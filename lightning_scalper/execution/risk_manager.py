#!/usr/bin/env python3
"""
🛡️ Lightning Scalper - Advanced Risk Management System
Production-Grade Risk Control for 80+ Client Trading Operations

This comprehensive risk management system provides multi-layered protection
for client accounts, position management, and system-wide risk control.

Features:
- Real-time position monitoring
- Dynamic risk limits per client
- Correlation-based exposure control
- Drawdown protection
- Emergency stop mechanisms
- News-based trading filters
- Market volatility monitoring
- Risk reporting and analytics

Author: Phoenix Trading AI (อาจารย์ฟินิกซ์)
Version: 1.0.0
License: Proprietary
"""

import asyncio
import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import our core modules
try:
    from core.lightning_scalper_engine import FVGSignal, CurrencyPair, FVGType
    from execution.trade_executor import ClientAccount, Position, Order
    from data.signal_logger import LightningScalperDataLogger
except ImportError as e:
    print(f"❌ Failed to import core modules: {e}")

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class RiskAction(Enum):
    """Risk management actions"""
    ALLOW = "ALLOW"
    REDUCE_SIZE = "REDUCE_SIZE"
    BLOCK_NEW = "BLOCK_NEW"
    CLOSE_PARTIAL = "CLOSE_PARTIAL"
    CLOSE_ALL = "CLOSE_ALL"
    EMERGENCY_STOP = "EMERGENCY_STOP"

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class RiskLimit:
    """Individual risk limit configuration"""
    name: str
    limit_type: str  # DAILY_LOSS, POSITION_SIZE, EXPOSURE, etc.
    threshold: float
    action: RiskAction
    is_active: bool = True
    alert_at_percent: float = 80.0  # Alert when 80% of limit reached
    
@dataclass
class RiskEvent:
    """Risk management event record"""
    timestamp: datetime
    client_id: str
    event_type: str
    risk_level: RiskLevel
    description: str
    action_taken: RiskAction
    limit_breached: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClientRiskProfile:
    """Complete risk profile for a client"""
    client_id: str
    account_balance: float
    
    # Risk limits
    max_daily_loss: float
    max_weekly_loss: float
    max_monthly_loss: float
    max_position_size: float
    max_total_exposure: float
    max_correlation_exposure: float
    max_positions_per_pair: int
    max_total_positions: int
    
    # Current status
    current_daily_pnl: float = 0.0
    current_weekly_pnl: float = 0.0
    current_monthly_pnl: float = 0.0
    current_exposure: float = 0.0
    current_positions: int = 0
    largest_position: float = 0.0
    
    # Risk multipliers based on performance
    performance_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    correlation_multiplier: float = 1.0
    
    # Status flags
    is_trading_enabled: bool = True
    risk_level: RiskLevel = RiskLevel.LOW
    last_risk_check: datetime = field(default_factory=datetime.now)
    
    # Violation tracking
    violation_count_today: int = 0
    last_violation: Optional[datetime] = None

@dataclass
class MarketRiskData:
    """Market-wide risk information"""
    volatility_index: float
    correlation_matrix: Dict[str, Dict[str, float]]
    major_news_events: List[Dict[str, Any]]
    market_hours_risk: Dict[str, float]
    spread_widening_pairs: List[str]
    unusual_volume_pairs: List[str]
    last_update: datetime = field(default_factory=datetime.now)

class LightningScalperRiskManager:
    """
    🛡️ Lightning Scalper Advanced Risk Management System
    Multi-layered protection for production trading operations
    """
    
    def __init__(self, data_logger: Optional[LightningScalperDataLogger] = None):
        self.data_logger = data_logger
        
        # Client risk profiles
        self.client_profiles: Dict[str, ClientRiskProfile] = {}
        self.risk_limits: Dict[str, List[RiskLimit]] = defaultdict(list)
        
        # Global risk settings
        self.global_limits = {
            'max_total_daily_loss': 10000.0,  # $10k across all clients
            'max_concurrent_trades': 200,      # Max trades system-wide
            'max_exposure_per_pair': 50.0,     # Max total exposure per currency pair
            'max_correlation_exposure': 30.0,  # Max exposure to correlated pairs
            'emergency_stop_threshold': 15000.0 # Emergency stop at $15k loss
        }
        
        # Market risk data
        self.market_risk = MarketRiskData(
            volatility_index=0.5,
            correlation_matrix={},
            major_news_events=[],
            market_hours_risk={},
            spread_widening_pairs=[],
            unusual_volume_pairs=[]
        )
        
        # Risk monitoring
        self.risk_events: deque = deque(maxlen=10000)
        self.alerts_queue: deque = deque(maxlen=1000)
        self.position_monitor: Dict[str, Dict] = {}
        
        # Real-time tracking
        self.total_system_pnl = 0.0
        self.total_system_exposure = 0.0
        self.active_positions_count = 0
        self.emergency_stop_active = False
        
        # Performance metrics
        self.stats = {
            'risk_checks_performed': 0,
            'signals_blocked': 0,
            'positions_closed': 0,
            'emergency_stops': 0,
            'alerts_generated': 0,
            'violations_detected': 0
        }
        
        # Threading for real-time monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.monitoring_interval = 5.0  # 5 seconds
        
        # Event callbacks
        self.risk_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Locks for thread safety
        self.risk_lock = threading.Lock()
        self.profiles_lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger('RiskManager')
        
        # Initialize correlation matrix
        self._initialize_correlation_matrix()
        
        self.logger.info("🛡️ Lightning Scalper Risk Manager initialized")
    
    def _initialize_correlation_matrix(self):
        """Initialize currency correlation matrix"""
        # Simplified correlation matrix for major pairs
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
        
        # Sample correlations (in production, calculate from historical data)
        correlations = {
            'EURUSD': {'GBPUSD': 0.65, 'USDJPY': -0.45, 'AUDUSD': 0.55, 'USDCAD': -0.75, 'USDCHF': -0.85, 'NZDUSD': 0.50},
            'GBPUSD': {'EURUSD': 0.65, 'USDJPY': -0.35, 'AUDUSD': 0.45, 'USDCAD': -0.60, 'USDCHF': -0.70, 'NZDUSD': 0.40},
            'USDJPY': {'EURUSD': -0.45, 'GBPUSD': -0.35, 'AUDUSD': -0.25, 'USDCAD': 0.35, 'USDCHF': 0.40, 'NZDUSD': -0.20},
            'AUDUSD': {'EURUSD': 0.55, 'GBPUSD': 0.45, 'USDJPY': -0.25, 'USDCAD': -0.50, 'USDCHF': -0.55, 'NZDUSD': 0.75},
            'USDCAD': {'EURUSD': -0.75, 'GBPUSD': -0.60, 'USDJPY': 0.35, 'AUDUSD': -0.50, 'USDCHF': 0.70, 'NZDUSD': -0.45},
            'USDCHF': {'EURUSD': -0.85, 'GBPUSD': -0.70, 'USDJPY': 0.40, 'AUDUSD': -0.55, 'USDCAD': 0.70, 'NZDUSD': -0.50},
            'NZDUSD': {'EURUSD': 0.50, 'GBPUSD': 0.40, 'USDJPY': -0.20, 'AUDUSD': 0.75, 'USDCAD': -0.45, 'USDCHF': -0.50}
        }
        
        # Build full matrix
        self.market_risk.correlation_matrix = {}
        for pair1 in pairs:
            self.market_risk.correlation_matrix[pair1] = {}
            for pair2 in pairs:
                if pair1 == pair2:
                    self.market_risk.correlation_matrix[pair1][pair2] = 1.0
                elif pair2 in correlations.get(pair1, {}):
                    self.market_risk.correlation_matrix[pair1][pair2] = correlations[pair1][pair2]
                elif pair1 in correlations.get(pair2, {}):
                    self.market_risk.correlation_matrix[pair1][pair2] = correlations[pair2][pair1]
                else:
                    self.market_risk.correlation_matrix[pair1][pair2] = 0.0
    
    def add_client(self, client_account: ClientAccount, custom_limits: Optional[Dict[str, float]] = None) -> bool:
        """Add client with risk profile"""
        try:
            with self.profiles_lock:
                # Create risk profile
                profile = ClientRiskProfile(
                    client_id=client_account.client_id,
                    account_balance=client_account.balance,
                    max_daily_loss=custom_limits.get('max_daily_loss', client_account.max_daily_loss),
                    max_weekly_loss=custom_limits.get('max_weekly_loss', client_account.max_weekly_loss),
                    max_monthly_loss=custom_limits.get('max_monthly_loss', client_account.max_monthly_loss),
                    max_position_size=custom_limits.get('max_position_size', client_account.max_lot_size * 100000),
                    max_total_exposure=custom_limits.get('max_total_exposure', client_account.balance * 0.20),
                    max_correlation_exposure=custom_limits.get('max_correlation_exposure', client_account.balance * 0.15),
                    max_positions_per_pair=custom_limits.get('max_positions_per_pair', 3),
                    max_total_positions=custom_limits.get('max_total_positions', client_account.max_positions)
                )
                
                self.client_profiles[client_account.client_id] = profile
                
                # Set up default risk limits
                self._setup_default_risk_limits(client_account.client_id)
                
                self.logger.info(f"🛡️ Risk profile created for client {client_account.client_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error adding client risk profile: {e}")
            return False
    
    def _setup_default_risk_limits(self, client_id: str):
        """Setup default risk limits for client"""
        profile = self.client_profiles[client_id]
        
        default_limits = [
            RiskLimit("daily_loss", "DAILY_LOSS", profile.max_daily_loss, RiskAction.BLOCK_NEW),
            RiskLimit("weekly_loss", "WEEKLY_LOSS", profile.max_weekly_loss, RiskAction.CLOSE_PARTIAL),
            RiskLimit("monthly_loss", "MONTHLY_LOSS", profile.max_monthly_loss, RiskAction.CLOSE_ALL),
            RiskLimit("position_size", "POSITION_SIZE", profile.max_position_size, RiskAction.REDUCE_SIZE),
            RiskLimit("total_exposure", "TOTAL_EXPOSURE", profile.max_total_exposure, RiskAction.BLOCK_NEW),
            RiskLimit("correlation_exposure", "CORRELATION_EXPOSURE", profile.max_correlation_exposure, RiskAction.REDUCE_SIZE),
            RiskLimit("max_positions", "MAX_POSITIONS", profile.max_total_positions, RiskAction.BLOCK_NEW)
        ]
        
        self.risk_limits[client_id] = default_limits
    
    def start_monitoring(self):
        """Start real-time risk monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("📊 Risk monitoring started")
    
    def stop_monitoring(self):
        """Stop risk monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=10)
            self.logger.info("🛑 Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Background risk monitoring loop"""
        while self.is_monitoring:
            try:
                # Perform risk checks
                self._perform_system_risk_check()
                
                # Update market risk data
                self._update_market_risk_data()
                
                # Process alerts
                self._process_risk_alerts()
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in risk monitoring loop: {e}")
                time.sleep(30)  # Longer sleep on error
    
    def _perform_system_risk_check(self):
        """Perform comprehensive system-wide risk check"""
        try:
            with self.risk_lock:
                self.stats['risk_checks_performed'] += 1
                
                # Check global limits
                self._check_global_limits()
                
                # Check individual client limits
                for client_id in self.client_profiles.keys():
                    self._check_client_limits(client_id)
                
                # Check correlation exposure
                self._check_correlation_exposure()
                
                # Update risk levels
                self._update_risk_levels()
                
        except Exception as e:
            self.logger.error(f"❌ Error in system risk check: {e}")
    
    def _check_global_limits(self):
        """Check system-wide risk limits"""
        # Check total daily loss
        if abs(self.total_system_pnl) > self.global_limits['max_total_daily_loss']:
            self._trigger_emergency_stop("Global daily loss limit exceeded")
            return
        
        # Check concurrent trades
        if self.active_positions_count > self.global_limits['max_concurrent_trades']:
            self._create_risk_event(
                "SYSTEM", "GLOBAL_LIMIT_BREACH", RiskLevel.HIGH,
                f"Concurrent trades limit exceeded: {self.active_positions_count}",
                RiskAction.BLOCK_NEW
            )
        
        # Check emergency stop threshold
        if abs(self.total_system_pnl) > self.global_limits['emergency_stop_threshold']:
            self._trigger_emergency_stop("Emergency stop threshold reached")
    
    def _check_client_limits(self, client_id: str):
        """Check individual client risk limits"""
        if client_id not in self.client_profiles:
            return
        
        profile = self.client_profiles[client_id]
        limits = self.risk_limits[client_id]
        
        for limit in limits:
            if not limit.is_active:
                continue
            
            current_value = self._get_current_limit_value(client_id, limit.limit_type)
            breach_percentage = (current_value / limit.threshold) * 100 if limit.threshold > 0 else 0
            
            # Check for limit breach
            if current_value >= limit.threshold:
                self._handle_limit_breach(client_id, limit, current_value)
            
            # Check for alert threshold
            elif breach_percentage >= limit.alert_at_percent:
                self._create_risk_alert(
                    client_id, AlertLevel.WARNING,
                    f"{limit.name} at {breach_percentage:.1f}% of limit",
                    {"current": current_value, "limit": limit.threshold}
                )
    
    def _get_current_limit_value(self, client_id: str, limit_type: str) -> float:
        """Get current value for specific limit type"""
        profile = self.client_profiles[client_id]
        
        if limit_type == "DAILY_LOSS":
            return abs(profile.current_daily_pnl) if profile.current_daily_pnl < 0 else 0
        elif limit_type == "WEEKLY_LOSS":
            return abs(profile.current_weekly_pnl) if profile.current_weekly_pnl < 0 else 0
        elif limit_type == "MONTHLY_LOSS":
            return abs(profile.current_monthly_pnl) if profile.current_monthly_pnl < 0 else 0
        elif limit_type == "POSITION_SIZE":
            return profile.largest_position
        elif limit_type == "TOTAL_EXPOSURE":
            return profile.current_exposure
        elif limit_type == "MAX_POSITIONS":
            return profile.current_positions
        else:
            return 0.0
    
    def _handle_limit_breach(self, client_id: str, limit: RiskLimit, current_value: float):
        """Handle risk limit breach"""
        profile = self.client_profiles[client_id]
        
        # Record violation
        profile.violation_count_today += 1
        profile.last_violation = datetime.now()
        
        # Create risk event
        risk_event = self._create_risk_event(
            client_id, "LIMIT_BREACH", RiskLevel.HIGH,
            f"{limit.name} limit breached: {current_value:.2f} > {limit.threshold:.2f}",
            limit.action,
            limit_breached=limit.name,
            current_value=current_value,
            limit_value=limit.threshold
        )
        
        # Execute risk action
        self._execute_risk_action(client_id, limit.action, risk_event)
        
        # Update risk level
        if profile.violation_count_today >= 3:
            profile.risk_level = RiskLevel.CRITICAL
            profile.is_trading_enabled = False
    
    def _execute_risk_action(self, client_id: str, action: RiskAction, risk_event: RiskEvent):
        """Execute specific risk management action"""
        try:
            if action == RiskAction.ALLOW:
                return
            
            elif action == RiskAction.REDUCE_SIZE:
                self._reduce_position_sizes(client_id)
                
            elif action == RiskAction.BLOCK_NEW:
                self._block_new_trades(client_id)
                
            elif action == RiskAction.CLOSE_PARTIAL:
                self._close_partial_positions(client_id)
                
            elif action == RiskAction.CLOSE_ALL:
                self._close_all_positions(client_id)
                
            elif action == RiskAction.EMERGENCY_STOP:
                self._trigger_emergency_stop(f"Risk action for client {client_id}")
            
            # Trigger callbacks
            self._trigger_risk_callbacks('risk_action_executed', {
                'client_id': client_id,
                'action': action,
                'risk_event': risk_event
            })
            
        except Exception as e:
            self.logger.error(f"❌ Error executing risk action {action} for {client_id}: {e}")
    
    def _reduce_position_sizes(self, client_id: str):
        """Reduce position sizes for client"""
        # This would integrate with trade executor to reduce position sizes
        self.logger.warning(f"⚠️ Reducing position sizes for client {client_id}")
        self.stats['positions_closed'] += 1
    
    def _block_new_trades(self, client_id: str):
        """Block new trades for client"""
        profile = self.client_profiles[client_id]
        profile.is_trading_enabled = False
        self.logger.warning(f"🚫 Blocking new trades for client {client_id}")
        self.stats['signals_blocked'] += 1
    
    def _close_partial_positions(self, client_id: str):
        """Close partial positions for client"""
        # This would integrate with trade executor to close positions
        self.logger.warning(f"📉 Closing partial positions for client {client_id}")
        self.stats['positions_closed'] += 1
    
    def _close_all_positions(self, client_id: str):
        """Close all positions for client"""
        # This would integrate with trade executor to close all positions
        self.logger.critical(f"🚨 Closing ALL positions for client {client_id}")
        self.stats['positions_closed'] += 1
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger system-wide emergency stop"""
        if not self.emergency_stop_active:
            self.emergency_stop_active = True
            self.stats['emergency_stops'] += 1
            
            # Disable trading for all clients
            with self.profiles_lock:
                for profile in self.client_profiles.values():
                    profile.is_trading_enabled = False
            
            # Create critical risk event
            self._create_risk_event(
                "SYSTEM", "EMERGENCY_STOP", RiskLevel.CRITICAL,
                f"Emergency stop triggered: {reason}",
                RiskAction.EMERGENCY_STOP
            )
            
            self.logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
            
            # Trigger callbacks
            self._trigger_risk_callbacks('emergency_stop', {'reason': reason})
    
    def evaluate_signal_risk(self, signal: FVGSignal, client_id: str, proposed_lot_size: float) -> Dict[str, Any]:
        """Evaluate risk for a proposed signal execution"""
        try:
            with self.risk_lock:
                self.stats['risk_checks_performed'] += 1
                
                if client_id not in self.client_profiles:
                    return {
                        'approved': False,
                        'reason': 'Client not found in risk profiles',
                        'risk_level': RiskLevel.CRITICAL,
                        'recommended_action': RiskAction.BLOCK_NEW
                    }
                
                profile = self.client_profiles[client_id]
                
                # Check if trading is enabled
                if not profile.is_trading_enabled:
                    return {
                        'approved': False,
                        'reason': 'Trading disabled for client',
                        'risk_level': RiskLevel.HIGH,
                        'recommended_action': RiskAction.BLOCK_NEW
                    }
                
                # Check emergency stop
                if self.emergency_stop_active:
                    return {
                        'approved': False,
                        'reason': 'System emergency stop active',
                        'risk_level': RiskLevel.CRITICAL,
                        'recommended_action': RiskAction.EMERGENCY_STOP
                    }
                
                # Calculate position risk
                position_value = proposed_lot_size * 100000  # Standard lot value
                risk_checks = []
                
                # 1. Position size check
                if position_value > profile.max_position_size:
                    risk_checks.append({
                        'type': 'POSITION_SIZE',
                        'passed': False,
                        'current': position_value,
                        'limit': profile.max_position_size
                    })
                
                # 2. Total exposure check
                new_exposure = profile.current_exposure + position_value
                if new_exposure > profile.max_total_exposure:
                    risk_checks.append({
                        'type': 'TOTAL_EXPOSURE',
                        'passed': False,
                        'current': new_exposure,
                        'limit': profile.max_total_exposure
                    })
                
                # 3. Correlation exposure check
                correlation_exposure = self._calculate_correlation_exposure(
                    client_id, signal.currency_pair.value, position_value
                )
                if correlation_exposure > profile.max_correlation_exposure:
                    risk_checks.append({
                        'type': 'CORRELATION_EXPOSURE',
                        'passed': False,
                        'current': correlation_exposure,
                        'limit': profile.max_correlation_exposure
                    })
                
                # 4. Signal quality check
                min_confluence = 60.0  # Minimum confluence score
                if signal.confluence_score < min_confluence:
                    risk_checks.append({
                        'type': 'SIGNAL_QUALITY',
                        'passed': False,
                        'current': signal.confluence_score,
                        'limit': min_confluence
                    })
                
                # 5. Market volatility check
                volatility_risk = self._assess_market_volatility_risk(signal.currency_pair.value)
                if volatility_risk > 0.8:  # High volatility
                    risk_checks.append({
                        'type': 'MARKET_VOLATILITY',
                        'passed': False,
                        'current': volatility_risk,
                        'limit': 0.8
                    })
                
                # Determine overall approval
                failed_checks = [check for check in risk_checks if not check['passed']]
                
                if not failed_checks:
                    # Apply performance multipliers
                    adjusted_lot_size = proposed_lot_size * profile.performance_multiplier
                    adjusted_lot_size *= profile.volatility_multiplier
                    adjusted_lot_size = max(0.01, min(adjusted_lot_size, profile.max_position_size / 100000))
                    
                    return {
                        'approved': True,
                        'original_lot_size': proposed_lot_size,
                        'adjusted_lot_size': adjusted_lot_size,
                        'risk_level': RiskLevel.LOW,
                        'risk_score': signal.confluence_score,
                        'recommended_action': RiskAction.ALLOW,
                        'risk_checks': risk_checks
                    }
                else:
                    # Determine severity
                    critical_failures = ['TOTAL_EXPOSURE', 'CORRELATION_EXPOSURE']
                    has_critical = any(check['type'] in critical_failures for check in failed_checks)
                    
                    risk_level = RiskLevel.CRITICAL if has_critical else RiskLevel.HIGH
                    action = RiskAction.BLOCK_NEW if has_critical else RiskAction.REDUCE_SIZE
                    
                    # Calculate reduced lot size
                    if action == RiskAction.REDUCE_SIZE:
                        reduction_factor = 0.5  # Reduce by 50%
                        adjusted_lot_size = proposed_lot_size * reduction_factor
                        
                        # Re-check with reduced size
                        if adjusted_lot_size >= 0.01:
                            return {
                                'approved': True,
                                'original_lot_size': proposed_lot_size,
                                'adjusted_lot_size': adjusted_lot_size,
                                'risk_level': risk_level,
                                'recommended_action': action,
                                'risk_checks': risk_checks,
                                'warnings': [f"Lot size reduced due to {check['type']}" for check in failed_checks]
                            }
                    
                    return {
                        'approved': False,
                        'reason': f"Failed risk checks: {[check['type'] for check in failed_checks]}",
                        'risk_level': risk_level,
                        'recommended_action': action,
                        'risk_checks': risk_checks
                    }
                    
        except Exception as e:
            self.logger.error(f"❌ Error evaluating signal risk: {e}")
            return {
                'approved': False,
                'reason': f'Risk evaluation error: {str(e)}',
                'risk_level': RiskLevel.CRITICAL,
                'recommended_action': RiskAction.BLOCK_NEW
            }
    
    def _calculate_correlation_exposure(self, client_id: str, currency_pair: str, new_position_value: float) -> float:
        """Calculate correlation-adjusted exposure"""
        try:
            profile = self.client_profiles[client_id]
            total_corr_exposure = new_position_value  # Start with new position
            
            # Get correlations for this pair
            correlations = self.market_risk.correlation_matrix.get(currency_pair, {})
            
            # Add correlation-weighted exposure from existing positions
            # This would integrate with actual position data
            # For now, estimate based on current exposure
            for other_pair, correlation in correlations.items():
                if abs(correlation) > 0.5:  # Only consider significant correlations
                    # Estimate exposure in this pair (simplified)
                    estimated_exposure = profile.current_exposure * 0.2  # Assume 20% in each major pair
                    total_corr_exposure += estimated_exposure * abs(correlation)
            
            return total_corr_exposure
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating correlation exposure: {e}")
            return new_position_value  # Return conservative estimate
    
    def _assess_market_volatility_risk(self, currency_pair: str) -> float:
        """Assess market volatility risk for currency pair"""
        try:
            # Base volatility risk
            base_risk = self.market_risk.volatility_index
            
            # Pair-specific adjustments
            volatile_pairs = ['GBPJPY', 'EURJPY', 'XAUUSD']
            if currency_pair in volatile_pairs:
                base_risk *= 1.3
            
            # Market hours adjustment
            current_hour = datetime.now().hour
            if 0 <= current_hour <= 6:  # Asian session (lower liquidity)
                base_risk *= 1.2
            elif 22 <= current_hour <= 23:  # Session overlap (higher volatility)
                base_risk *= 1.1
            
            # News events adjustment
            if self._has_major_news_today(currency_pair):
                base_risk *= 1.5
            
            return min(base_risk, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"❌ Error assessing volatility risk: {e}")
            return 0.5  # Return moderate risk as default
    
    def _has_major_news_today(self, currency_pair: str) -> bool:
        """Check if there are major news events today for currency pair"""
        today = datetime.now().date()
        
        for event in self.market_risk.major_news_events:
            event_date = event.get('date')
            if isinstance(event_date, str):
                event_date = datetime.fromisoformat(event_date).date()
            
            if (event_date == today and 
                currency_pair[:3] in event.get('currencies', []) or
                currency_pair[3:] in event.get('currencies', [])):
                return True
        
        return False
    
    def update_client_pnl(self, client_id: str, pnl_change: float, period: str = 'daily'):
        """Update client P&L and check limits"""
        try:
            with self.profiles_lock:
                if client_id not in self.client_profiles:
                    return
                
                profile = self.client_profiles[client_id]
                
                if period == 'daily':
                    profile.current_daily_pnl += pnl_change
                elif period == 'weekly':
                    profile.current_weekly_pnl += pnl_change
                elif period == 'monthly':
                    profile.current_monthly_pnl += pnl_change
                
                # Update system totals
                self.total_system_pnl += pnl_change
                
                # Trigger risk check
                self._check_client_limits(client_id)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating client P&L: {e}")
    
    def update_client_positions(self, client_id: str, positions_data: Dict[str, Any]):
        """Update client position information"""
        try:
            with self.profiles_lock:
                if client_id not in self.client_profiles:
                    return
                
                profile = self.client_profiles[client_id]
                
                # Update position metrics
                profile.current_positions = positions_data.get('count', 0)
                profile.current_exposure = positions_data.get('total_exposure', 0.0)
                profile.largest_position = positions_data.get('largest_position', 0.0)
                profile.last_risk_check = datetime.now()
                
                # Update system totals
                self.active_positions_count = sum(
                    p.current_positions for p in self.client_profiles.values()
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error updating client positions: {e}")
    
    def _check_correlation_exposure(self):
        """Check system-wide correlation exposure"""
        try:
            # Calculate total exposure per currency pair
            pair_exposures = defaultdict(float)
            
            for profile in self.client_profiles.values():
                # This would integrate with real position data
                # For now, estimate exposure distribution
                estimated_per_pair = profile.current_exposure / 5  # Assume 5 major pairs
                for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD']:
                    pair_exposures[pair] += estimated_per_pair
            
            # Check for excessive exposure in correlated pairs
            for pair1, exposure1 in pair_exposures.items():
                if exposure1 > self.global_limits['max_exposure_per_pair']:
                    self._create_risk_event(
                        "SYSTEM", "PAIR_EXPOSURE_LIMIT", RiskLevel.HIGH,
                        f"Excessive exposure in {pair1}: ${exposure1:.2f}",
                        RiskAction.REDUCE_SIZE
                    )
                
                # Check correlated pairs
                correlations = self.market_risk.correlation_matrix.get(pair1, {})
                for pair2, correlation in correlations.items():
                    if abs(correlation) > 0.7 and pair2 in pair_exposures:
                        combined_exposure = exposure1 + (pair_exposures[pair2] * abs(correlation))
                        
                        if combined_exposure > self.global_limits['max_correlation_exposure']:
                            self._create_risk_event(
                                "SYSTEM", "CORRELATION_EXPOSURE", RiskLevel.MEDIUM,
                                f"High correlation exposure: {pair1}-{pair2} = ${combined_exposure:.2f}",
                                RiskAction.BLOCK_NEW
                            )
                            
        except Exception as e:
            self.logger.error(f"❌ Error checking correlation exposure: {e}")
    
    def _update_risk_levels(self):
        """Update risk levels for all clients"""
        try:
            for client_id, profile in self.client_profiles.items():
                # Calculate overall risk score
                risk_score = 0
                
                # P&L risk
                daily_loss_ratio = abs(profile.current_daily_pnl) / profile.max_daily_loss if profile.max_daily_loss > 0 else 0
                risk_score += daily_loss_ratio * 30
                
                # Exposure risk
                exposure_ratio = profile.current_exposure / profile.max_total_exposure if profile.max_total_exposure > 0 else 0
                risk_score += exposure_ratio * 25
                
                # Position count risk
                position_ratio = profile.current_positions / profile.max_total_positions if profile.max_total_positions > 0 else 0
                risk_score += position_ratio * 20
                
                # Violation penalty
                risk_score += profile.violation_count_today * 10
                
                # Market volatility impact
                risk_score += self.market_risk.volatility_index * 15
                
                # Determine risk level
                if risk_score >= 80:
                    profile.risk_level = RiskLevel.CRITICAL
                elif risk_score >= 60:
                    profile.risk_level = RiskLevel.HIGH
                elif risk_score >= 40:
                    profile.risk_level = RiskLevel.MEDIUM
                elif risk_score >= 20:
                    profile.risk_level = RiskLevel.LOW
                else:
                    profile.risk_level = RiskLevel.VERY_LOW
                
                # Update performance multipliers
                self._update_performance_multipliers(client_id, risk_score)
                
        except Exception as e:
            self.logger.error(f"❌ Error updating risk levels: {e}")
    
    def _update_performance_multipliers(self, client_id: str, risk_score: float):
        """Update performance-based multipliers"""
        profile = self.client_profiles[client_id]
        
        # Performance multiplier based on recent P&L
        if profile.current_daily_pnl > 0:
            profile.performance_multiplier = min(1.2, 1.0 + (profile.current_daily_pnl / profile.account_balance))
        else:
            profile.performance_multiplier = max(0.5, 1.0 + (profile.current_daily_pnl / profile.account_balance))
        
        # Volatility multiplier
        profile.volatility_multiplier = max(0.8, 1.2 - self.market_risk.volatility_index)
        
        # Risk-based adjustment
        if risk_score > 60:
            profile.performance_multiplier *= 0.8
            profile.volatility_multiplier *= 0.9
    
    def _update_market_risk_data(self):
        """Update market risk data (news, volatility, etc.)"""
        try:
            # Update volatility index (simplified)
            base_volatility = 0.5
            current_hour = datetime.now().hour
            
            # Session-based volatility adjustment
            if 13 <= current_hour <= 16:  # London-NY overlap
                base_volatility *= 1.3
            elif 8 <= current_hour <= 12:  # London session
                base_volatility *= 1.1
            elif 0 <= current_hour <= 6:  # Asian session
                base_volatility *= 0.8
            
            self.market_risk.volatility_index = base_volatility
            self.market_risk.last_update = datetime.now()
            
            # Update news events (in production, fetch from news API)
            self._update_news_events()
            
        except Exception as e:
            self.logger.error(f"❌ Error updating market risk data: {e}")
    
    def _update_news_events(self):
        """Update major news events (placeholder)"""
        # In production, this would fetch from economic calendar API
        today = datetime.now().date()
        
        # Sample news events
        sample_events = [
            {
                'date': today.isoformat(),
                'time': '14:30',
                'event': 'US Non-Farm Payrolls',
                'impact': 'HIGH',
                'currencies': ['USD'],
                'forecast': '200K',
                'previous': '180K'
            },
            {
                'date': today.isoformat(),
                'time': '12:30',
                'event': 'ECB Interest Rate Decision',
                'impact': 'HIGH',
                'currencies': ['EUR'],
                'forecast': '4.50%',
                'previous': '4.50%'
            }
        ]
        
        self.market_risk.major_news_events = sample_events
    
    def _create_risk_event(self, client_id: str, event_type: str, risk_level: RiskLevel,
                          description: str, action_taken: RiskAction, **kwargs) -> RiskEvent:
        """Create and store risk event"""
        event = RiskEvent(
            timestamp=datetime.now(),
            client_id=client_id,
            event_type=event_type,
            risk_level=risk_level,
            description=description,
            action_taken=action_taken,
            **kwargs
        )
        
        self.risk_events.append(event)
        self.stats['violations_detected'] += 1
        
        # Log to data logger if available
        if self.data_logger:
            try:
                self.data_logger.log_performance({
                    'timestamp': event.timestamp,
                    'client_id': client_id,
                    'period_type': 'RISK_EVENT',
                    'total_pnl': kwargs.get('current_value', 0),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'max_drawdown': 0,
                    'signals_generated': 0,
                    'signals_executed': 0,
                    'execution_rate': 0,
                    'avg_signal_quality': 0
                })
            except Exception as e:
                self.logger.error(f"❌ Error logging risk event: {e}")
        
        return event
    
    def _create_risk_alert(self, client_id: str, level: AlertLevel, message: str, metadata: Dict[str, Any]):
        """Create risk alert"""
        alert = {
            'timestamp': datetime.now(),
            'client_id': client_id,
            'level': level,
            'message': message,
            'metadata': metadata
        }
        
        self.alerts_queue.append(alert)
        self.stats['alerts_generated'] += 1
        
        self.logger.warning(f"⚠️ Risk Alert [{level.value}] {client_id}: {message}")
    
    def _process_risk_alerts(self):
        """Process pending risk alerts"""
        # Process alerts in queue (send notifications, etc.)
        while self.alerts_queue:
            alert = self.alerts_queue.popleft()
            
            # In production, send notifications here
            # (email, Telegram, dashboard updates, etc.)
            
            # Trigger callbacks
            self._trigger_risk_callbacks('risk_alert', alert)
    
    def _trigger_risk_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Trigger risk management callbacks"""
        try:
            callbacks = self.risk_callbacks.get(event_type, [])
            for callback in callbacks:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.error(f"❌ Error in risk callback: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error triggering risk callbacks: {e}")
    
    def subscribe_to_risk_events(self, event_type: str, callback: Callable):
        """Subscribe to risk management events"""
        self.risk_callbacks[event_type].append(callback)
        self.logger.debug(f"📡 Subscribed to risk event: {event_type}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk management summary"""
        try:
            with self.risk_lock:
                # Client risk summary
                client_risks = {}
                for client_id, profile in self.client_profiles.items():
                    client_risks[client_id] = {
                        'risk_level': profile.risk_level.value,
                        'trading_enabled': profile.is_trading_enabled,
                        'daily_pnl': profile.current_daily_pnl,
                        'exposure': profile.current_exposure,
                        'positions': profile.current_positions,
                        'violations_today': profile.violation_count_today,
                        'performance_multiplier': profile.performance_multiplier
                    }
                
                # Global risk metrics
                total_clients = len(self.client_profiles)
                active_clients = sum(1 for p in self.client_profiles.values() if p.is_trading_enabled)
                high_risk_clients = sum(1 for p in self.client_profiles.values() if p.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
                
                # Recent events
                recent_events = list(self.risk_events)[-10:] if self.risk_events else []
                
                return {
                    'global_metrics': {
                        'total_system_pnl': self.total_system_pnl,
                        'total_exposure': self.total_system_exposure,
                        'active_positions': self.active_positions_count,
                        'emergency_stop_active': self.emergency_stop_active,
                        'market_volatility': self.market_risk.volatility_index
                    },
                    'client_metrics': {
                        'total_clients': total_clients,
                        'active_clients': active_clients,
                        'high_risk_clients': high_risk_clients,
                        'risk_distribution': {
                            level.value: sum(1 for p in self.client_profiles.values() if p.risk_level == level)
                            for level in RiskLevel
                        }
                    },
                    'statistics': self.stats,
                    'recent_events': [
                        {
                            'timestamp': event.timestamp.isoformat(),
                            'client_id': event.client_id,
                            'type': event.event_type,
                            'level': event.risk_level.value,
                            'description': event.description,
                            'action': event.action_taken.value
                        }
                        for event in recent_events
                    ],
                    'client_details': client_risks,
                    'market_risk': {
                        'volatility_index': self.market_risk.volatility_index,
                        'major_news_today': len(self.market_risk.major_news_events),
                        'last_update': self.market_risk.last_update.isoformat()
                    }
                }
                
        except Exception as e:
            self.logger.error(f"❌ Error generating risk summary: {e}")
            return {'error': str(e)}
    
    def reset_daily_limits(self):
        """Reset daily risk limits (call at start of each trading day)"""
        try:
            with self.profiles_lock:
                for profile in self.client_profiles.values():
                    profile.current_daily_pnl = 0.0
                    profile.violation_count_today = 0
                    profile.is_trading_enabled = True
                    profile.risk_level = RiskLevel.LOW
                
                # Reset global metrics
                self.total_system_pnl = 0.0
                self.emergency_stop_active = False
                
                self.logger.info("🔄 Daily risk limits reset")
                
        except Exception as e:
            self.logger.error(f"❌ Error resetting daily limits: {e}")
    
    def enable_emergency_mode(self):
        """Enable emergency trading mode with stricter limits"""
        try:
            with self.profiles_lock:
                for profile in self.client_profiles.values():
                    # Reduce all limits by 50%
                    profile.max_daily_loss *= 0.5
                    profile.max_position_size *= 0.5
                    profile.max_total_exposure *= 0.5
                    profile.performance_multiplier *= 0.5
                    profile.volatility_multiplier *= 0.8
                
                self.logger.warning("🚨 Emergency trading mode enabled - All limits reduced")
                
        except Exception as e:
            self.logger.error(f"❌ Error enabling emergency mode: {e}")


# Demo and testing functionality
def demo_risk_manager():
    """Demonstrate the risk management system"""
    print("🛡️ Lightning Scalper Risk Manager - Demo")
    print("=" * 60)
    
    # Initialize risk manager
    risk_manager = LightningScalperRiskManager()
    
    # Create sample clients
    from execution.trade_executor import ClientAccount
    
    sample_clients = [
        ClientAccount(
            client_id="RISK_TEST_001",
            account_number="12345001",
            broker="TestBroker",
            currency="USD",
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            free_margin=10000.0,
            margin_level=0.0,
            max_daily_loss=200.0,
            max_weekly_loss=500.0,
            max_monthly_loss=1500.0,
            max_positions=5,
            max_lot_size=1.0
        ),
        ClientAccount(
            client_id="RISK_TEST_002",
            account_number="12345002",
            broker="TestBroker",
            currency="USD",
            balance=25000.0,
            equity=25000.0,
            margin=0.0,
            free_margin=25000.0,
            margin_level=0.0,
            max_daily_loss=500.0,
            max_weekly_loss=1200.0,
            max_monthly_loss=3000.0,
            max_positions=8,
            max_lot_size=2.0
        )
    ]
    
    # Add clients to risk manager
    for client in sample_clients:
        success = risk_manager.add_client(client)
        print(f"✅ Added client {client.client_id}: {success}")
    
    # Start monitoring
    risk_manager.start_monitoring()
    print("📊 Risk monitoring started")
    
    # Test signal evaluation
    from core.lightning_scalper_engine import FVGSignal, FVGType, CurrencyPair, MarketCondition, FVGStatus
    
    test_signal = FVGSignal(
        id="RISK_TEST_SIGNAL_001",
        timestamp=datetime.now(),
        timeframe="M5",
        currency_pair=CurrencyPair.EURUSD,
        fvg_type=FVGType.BULLISH,
        high=1.1050,
        low=1.1030,
        gap_size=0.0020,
        gap_percentage=0.18,
        confluence_score=75.0,
        market_condition=MarketCondition.TRENDING_UP,
        session="London",
        status=FVGStatus.ACTIVE,
        entry_price=1.1045,
        target_1=1.1065,
        target_2=1.1075,
        target_3=1.1085,
        stop_loss=1.1025,
        risk_reward_ratio=1.5,
        position_size_factor=1.0,
        urgency_level=3,
        atr_ratio=1.2,
        volume_strength=20.0,
        momentum_score=15.0,
        structure_score=40.0
    )
    
    # Evaluate signal risk for each client
    for client in sample_clients:
        print(f"\n🎯 Testing signal evaluation for {client.client_id}:")
        
        risk_result = risk_manager.evaluate_signal_risk(test_signal, client.client_id, 0.1)
        
        print(f"   Approved: {risk_result['approved']}")
        print(f"   Risk Level: {risk_result['risk_level'].value}")
        print(f"   Recommended Action: {risk_result['recommended_action'].value}")
        
        if risk_result['approved']:
            print(f"   Original Lot Size: {risk_result.get('original_lot_size', 'N/A')}")
            print(f"   Adjusted Lot Size: {risk_result.get('adjusted_lot_size', 'N/A')}")
        else:
            print(f"   Reason: {risk_result.get('reason', 'N/A')}")
    
    # Simulate some P&L changes
    print(f"\n📊 Simulating P&L changes...")
    risk_manager.update_client_pnl("RISK_TEST_001", -150.0)  # Losing trade
    risk_manager.update_client_pnl("RISK_TEST_002", 75.0)    # Winning trade
    
    # Update position information
    risk_manager.update_client_positions("RISK_TEST_001", {
        'count': 3,
        'total_exposure': 3000.0,
        'largest_position': 1200.0
    })
    
    # Wait for monitoring cycle
    time.sleep(6)
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    
    print(f"\n📈 Risk Management Summary:")
    print(f"   Total System P&L: ${summary['global_metrics']['total_system_pnl']:.2f}")
    print(f"   Active Clients: {summary['client_metrics']['active_clients']}/{summary['client_metrics']['total_clients']}")
    print(f"   High Risk Clients: {summary['client_metrics']['high_risk_clients']}")
    print(f"   Risk Checks Performed: {summary['statistics']['risk_checks_performed']}")
    print(f"   Alerts Generated: {summary['statistics']['alerts_generated']}")
    
    print(f"\n👥 Client Risk Status:")
    for client_id, details in summary['client_details'].items():
        print(f"   {client_id}:")
        print(f"     Risk Level: {details['risk_level']}")
        print(f"     Trading Enabled: {details['trading_enabled']}")
        print(f"     Daily P&L: ${details['daily_pnl']:.2f}")
        print(f"     Current Exposure: ${details['exposure']:.2f}")
    
    # Test emergency scenario
    print(f"\n🚨 Testing Emergency Scenario...")
    risk_manager.update_client_pnl("RISK_TEST_001", -300.0)  # Large loss triggering limit
    
    time.sleep(2)
    
    # Final summary
    final_summary = risk_manager.get_risk_summary()
    print(f"\n📊 Final Status:")
    print(f"   Emergency Stop: {final_summary['global_metrics']['emergency_stop_active']}")
    print(f"   Recent Events: {len(final_summary['recent_events'])}")
    
    if final_summary['recent_events']:
        print(f"   Last Event: {final_summary['recent_events'][-1]['description']}")
    
    # Stop monitoring
    risk_manager.stop_monitoring()
    print(f"\n✅ Risk Manager Demo Complete!")


if __name__ == "__main__":
    demo_risk_manager()