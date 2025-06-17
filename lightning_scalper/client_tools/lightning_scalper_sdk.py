#!/usr/bin/env python3
"""
🚀 Lightning Scalper - Python Client SDK
Production-Grade Client Integration Library

This SDK allows clients to integrate with the Lightning Scalper system,
receive trading signals, execute trades, and send performance data back
to the central system for active learning.

📍 ไฟล์นี้: client_tools/lightning_scalper_sdk.py

Features:
- Real-time signal reception
- Automated trade execution
- Performance tracking
- Risk management
- Data synchronization
- Multi-broker support

Author: Phoenix Trading AI (อาจารย์ฟินิกซ์)
Version: 1.0.0
License: Proprietary
"""

import asyncio
import json
import logging
import threading
import time
import websocket
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hmac
import hashlib
import base64

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class TradingSignal:
    """Lightning Scalper Trading Signal"""
    signal_id: str
    timestamp: datetime
    currency_pair: str
    signal_type: SignalType
    
    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Signal quality
    confluence_score: float
    confidence: float
    
    # Risk management
    risk_reward_ratio: float
    suggested_lot_size: float
    max_risk_percent: float
    
    # Metadata
    gap_size: float
    timeframe: str
    session: str
    market_condition: str
    
    # Expiry
    expires_at: datetime
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TradingSignal':
        """Create signal from dictionary"""
        return cls(
            signal_id=data['signal_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            currency_pair=data['currency_pair'],
            signal_type=SignalType(data['signal_type']),
            entry_price=data['entry_price'],
            stop_loss=data['stop_loss'],
            take_profit_1=data['take_profit_1'],
            take_profit_2=data['take_profit_2'],
            take_profit_3=data['take_profit_3'],
            confluence_score=data['confluence_score'],
            confidence=data['confidence'],
            risk_reward_ratio=data['risk_reward_ratio'],
            suggested_lot_size=data['suggested_lot_size'],
            max_risk_percent=data['max_risk_percent'],
            gap_size=data['gap_size'],
            timeframe=data['timeframe'],
            session=data['session'],
            market_condition=data['market_condition'],
            expires_at=datetime.fromisoformat(data['expires_at'])
        )

@dataclass
class TradeResult:
    """Trade execution result"""
    signal_id: str
    client_id: str
    timestamp: datetime
    
    # Execution details
    executed: bool
    execution_price: float
    lot_size: float
    slippage: float
    execution_delay: float
    
    # Results
    closed_at: Optional[datetime] = None
    close_price: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_pips: Optional[float] = None
    
    # Hit targets
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    sl_hit: bool = False
    
    # Additional metrics
    max_drawdown: float = 0.0
    max_profit: float = 0.0
    duration_minutes: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API submission"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['timestamp'] = self.timestamp.isoformat()
        if self.closed_at:
            data['closed_at'] = self.closed_at.isoformat()
        return data

class LightningScalperClient:
    """
    🚀 Lightning Scalper Client SDK
    Main client class for integration with Lightning Scalper system
    """
    
    def __init__(self, 
                 client_id: str,
                 api_key: str,
                 api_secret: str,
                 server_url: str = "ws://localhost:8080",
                 http_url: str = "http://localhost:5000"):
        
        self.client_id = client_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.server_url = server_url
        self.http_url = http_url
        
        # Connection state
        self.ws: Optional[websocket.WebSocketApp] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # Signal handling
        self.signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self.active_signals: Dict[str, TradingSignal] = {}
        self.trade_results: List[TradeResult] = []
        
        # Threading
        self.lock = threading.Lock()
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f'LightningClient-{client_id}')
        
        # Performance tracking
        self.stats = {
            'signals_received': 0,
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_pips': 0.0,
            'win_rate': 0.0,
            'connection_uptime': 0.0
        }
        
        self.logger.info(f"🚀 Lightning Scalper Client {client_id} initialized")
    
    def _generate_signature(self, message: str) -> str:
        """Generate HMAC signature for authentication"""
        return base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
    
    def _authenticate(self) -> dict:
        """Generate authentication payload"""
        timestamp = str(int(time.time()))
        message = f"{self.client_id}{timestamp}"
        signature = self._generate_signature(message)
        
        return {
            'client_id': self.client_id,
            'api_key': self.api_key,
            'timestamp': timestamp,
            'signature': signature
        }
    
    def add_signal_callback(self, callback: Callable[[TradingSignal], None]):
        """Add callback for signal reception"""
        self.signal_callbacks.append(callback)
        self.logger.info(f"Signal callback added: {callback.__name__}")
    
    def connect(self) -> bool:
        """Connect to Lightning Scalper server"""
        try:
            def on_message(ws, message):
                self._handle_message(message)
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.warning("WebSocket connection closed")
                self.is_connected = False
                self._attempt_reconnect()
            
            def on_open(ws):
                self.logger.info("WebSocket connection opened")
                self.is_connected = True
                self.reconnect_attempts = 0
                
                # Send authentication
                auth_data = self._authenticate()
                ws.send(json.dumps({
                    'type': 'auth',
                    'data': auth_data
                }))
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start connection in separate thread
            self.is_running = True
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Start heartbeat
            self._start_heartbeat()
            
            # Wait for connection
            time.sleep(2)
            return self.is_connected
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    def _handle_message(self, message: str):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'auth_success':
                self.logger.info("Authentication successful")
                
            elif msg_type == 'auth_failed':
                self.logger.error("Authentication failed")
                self.disconnect()
                
            elif msg_type == 'new_signal':
                self._handle_new_signal(data['data'])
                
            elif msg_type == 'signal_update':
                self._handle_signal_update(data['data'])
                
            elif msg_type == 'heartbeat':
                self._send_heartbeat_response()
                
            elif msg_type == 'system_message':
                self.logger.info(f"System message: {data['data']['message']}")
                
            else:
                self.logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle message: {e}")
    
    def _handle_new_signal(self, signal_data: dict):
        """Handle new trading signal"""
        try:
            signal = TradingSignal.from_dict(signal_data)
            
            with self.lock:
                self.active_signals[signal.signal_id] = signal
                self.stats['signals_received'] += 1
            
            self.logger.info(f"📡 New signal: {signal.currency_pair} {signal.signal_type.value} @ {signal.entry_price}")
            
            # Notify callbacks
            for callback in self.signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"Signal callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to handle new signal: {e}")
    
    def _handle_signal_update(self, update_data: dict):
        """Handle signal update"""
        signal_id = update_data.get('signal_id')
        if signal_id in self.active_signals:
            self.logger.info(f"Signal update for {signal_id}: {update_data}")
    
    def _start_heartbeat(self):
        """Start heartbeat thread"""
        def heartbeat_worker():
            while self.is_running and self.is_connected:
                try:
                    if self.ws:
                        self.ws.send(json.dumps({
                            'type': 'heartbeat',
                            'client_id': self.client_id,
                            'timestamp': datetime.now().isoformat()
                        }))
                    time.sleep(30)  # Send heartbeat every 30 seconds
                except Exception as e:
                    self.logger.error(f"Heartbeat error: {e}")
                    break
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
    
    def _send_heartbeat_response(self):
        """Send heartbeat response"""
        if self.ws:
            self.ws.send(json.dumps({
                'type': 'heartbeat_response',
                'client_id': self.client_id,
                'timestamp': datetime.now().isoformat()
            }))
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to server"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            self.logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        
        self.logger.info(f"Reconnection attempt {self.reconnect_attempts} in {wait_time}s")
        time.sleep(wait_time)
        
        self.connect()
    
    def submit_trade_result(self, trade_result: TradeResult) -> bool:
        """Submit trade result to server for active learning"""
        try:
            # Add authentication headers
            headers = {
                'Content-Type': 'application/json',
                'X-Client-ID': self.client_id,
                'X-API-Key': self.api_key
            }
            
            # Add signature
            timestamp = str(int(time.time()))
            message = f"{self.client_id}{timestamp}{json.dumps(trade_result.to_dict())}"
            headers['X-Timestamp'] = timestamp
            headers['X-Signature'] = self._generate_signature(message)
            
            # Submit to server
            response = requests.post(
                f"{self.http_url}/api/v1/trade-results",
                json=trade_result.to_dict(),
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info(f"Trade result submitted: {trade_result.signal_id}")
                
                with self.lock:
                    self.trade_results.append(trade_result)
                    self.stats['trades_executed'] += 1
                    if trade_result.profit_loss and trade_result.profit_loss > 0:
                        self.stats['successful_trades'] += 1
                    else:
                        self.stats['failed_trades'] += 1
                    
                    # Update win rate
                    if self.stats['trades_executed'] > 0:
                        self.stats['win_rate'] = self.stats['successful_trades'] / self.stats['trades_executed']
                
                return True
            else:
                self.logger.error(f"Failed to submit trade result: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error submitting trade result: {e}")
            return False
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get currently active signals"""
        with self.lock:
            return list(self.active_signals.values())
    
    def get_signal_by_id(self, signal_id: str) -> Optional[TradingSignal]:
        """Get signal by ID"""
        with self.lock:
            return self.active_signals.get(signal_id)
    
    def get_client_stats(self) -> dict:
        """Get client statistics"""
        with self.lock:
            return self.stats.copy()
    
    def get_account_info(self) -> dict:
        """Get account information from server"""
        try:
            headers = {
                'X-Client-ID': self.client_id,
                'X-API-Key': self.api_key
            }
            
            timestamp = str(int(time.time()))
            message = f"{self.client_id}{timestamp}"
            headers['X-Timestamp'] = timestamp
            headers['X-Signature'] = self._generate_signature(message)
            
            response = requests.get(
                f"{self.http_url}/api/v1/account-info",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get account info: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def update_account_balance(self, balance: float, equity: float, margin: float):
        """Update account balance information"""
        try:
            data = {
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'timestamp': datetime.now().isoformat()
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Client-ID': self.client_id,
                'X-API-Key': self.api_key
            }
            
            timestamp = str(int(time.time()))
            message = f"{self.client_id}{timestamp}{json.dumps(data)}"
            headers['X-Timestamp'] = timestamp
            headers['X-Signature'] = self._generate_signature(message)
            
            response = requests.post(
                f"{self.http_url}/api/v1/account-update",
                json=data,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.debug("Account balance updated")
                return True
            else:
                self.logger.error(f"Failed to update account balance: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating account balance: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from server"""
        self.is_running = False
        self.is_connected = False
        
        if self.ws:
            self.ws.close()
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=5)
        
        self.logger.info("Disconnected from Lightning Scalper server")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

# Example usage and utilities
def create_demo_client(client_id: str, api_key: str, api_secret: str) -> LightningScalperClient:
    """Create a demo client instance"""
    return LightningScalperClient(
        client_id=client_id,
        api_key=api_key,
        api_secret=api_secret,
        server_url="ws://demo.lightning-scalper.com:8080",
        http_url="https://demo.lightning-scalper.com"
    )

def create_live_client(client_id: str, api_key: str, api_secret: str) -> LightningScalperClient:
    """Create a live client instance"""
    return LightningScalperClient(
        client_id=client_id,
        api_key=api_key,
        api_secret=api_secret,
        server_url="wss://live.lightning-scalper.com:8080",
        http_url="https://live.lightning-scalper.com"
    )

if __name__ == "__main__":
    # Example usage
    print("🚀 Lightning Scalper Python Client SDK")
    print("This is a library file. Import it in your trading application.")
    print("\nExample usage:")
    print("""
    from lightning_scalper_sdk import LightningScalperClient, TradingSignal
    
    # Create client
    client = LightningScalperClient(
        client_id="your_client_id",
        api_key="your_api_key",
        api_secret="your_api_secret"
    )
    
    # Add signal handler
    def handle_signal(signal: TradingSignal):
        print(f"New signal: {signal.currency_pair} {signal.signal_type}")
        # Your trading logic here
    
    client.add_signal_callback(handle_signal)
    
    # Connect and run
    with client:
        # Client will receive signals automatically
        input("Press Enter to exit...")
    """)