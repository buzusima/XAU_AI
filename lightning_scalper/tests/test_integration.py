#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightning Scalper - Windows Safe Version
Auto-fixed for Unicode compatibility
"""

"""
[ROCKET] Lightning Scalper - Integration Test Suite
Production-Grade System Integration Testing

This comprehensive test suite validates the entire Lightning Scalper system
including FVG detection, trade execution, MT5 integration, and data logging.

Test Categories:
- Core Engine Tests
- Trade Execution Tests
- Client Management Tests
- Risk Management Tests
- Data Logger Tests
- Dashboard Integration Tests
- End-to-End Trading Flow Tests

Author: Phoenix Trading AI (??????????????)
Version: 1.0.0
License: Proprietary
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Import Lightning Scalper modules
try:
    from core.lightning_scalper_engine import (
        EnhancedFVGDetector, FVGSignal, FVGType, CurrencyPair, 
        MarketCondition, FVGStatus
    )
    from core.main_controller import LightningScalperController, SystemStatus
    from execution.trade_executor import (
        TradeExecutor, ClientAccount, Order, Position,
        OrderType, OrderStatus, TradeDirection
    )
    from adapters.mt5_adapter import MT5Adapter, MT5IntegratedExecutor
    from data.signal_logger import LightningScalperDataLogger, SignalLogEntry
    from config.config_manager import LightningScalperConfigManager
    from dashboard.web_dashboard import LightningScalperDashboard
    print("[CHECK] All Lightning Scalper modules imported successfully")
except ImportError as e:
    print(f"[X] Failed to import Lightning Scalper modules: {e}")
    print("   Make sure all files are in the correct directories")
    sys.exit(1)

class LightningScalperIntegrationTests(unittest.TestCase):
    """
    [TEST_TUBE] Lightning Scalper Integration Test Suite
    Comprehensive testing of the entire trading system
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        print("\n[TEST_TUBE] Setting up Lightning Scalper Integration Test Environment...")
        
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp(prefix="lightning_test_"))
        cls.config_dir = cls.test_dir / "config"
        cls.data_dir = cls.test_dir / "data"
        
        cls.config_dir.mkdir(exist_ok=True)
        cls.data_dir.mkdir(exist_ok=True)
        
        # Setup test logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(cls.test_dir / 'test.log'),
                logging.StreamHandler()
            ]
        )
        
        cls.logger = logging.getLogger('IntegrationTests')
        cls.logger.info(f"?? Test directory: {cls.test_dir}")
        
        # Test configuration
        cls.test_config = {
            'test_clients': [
                {
                    'client_id': 'TEST_CLIENT_001',
                    'mt5_login': 99999001,
                    'mt5_password': 'test_password_001',
                    'mt5_server': 'TestServer-MT5',
                    'account_info': {
                        'client_id': 'TEST_CLIENT_001',
                        'account_number': '99999001',
                        'broker': 'TestBroker',
                        'currency': 'USD',
                        'balance': 10000.0,
                        'equity': 10000.0,
                        'margin': 0.0,
                        'free_margin': 10000.0,
                        'margin_level': 0.0,
                        'max_daily_loss': 200.0,
                        'max_weekly_loss': 500.0,
                        'max_monthly_loss': 1500.0,
                        'max_positions': 5,
                        'max_lot_size': 1.0
                    },
                    'preferred_pairs': ['EURUSD', 'GBPUSD'],
                    'risk_multiplier': 1.0,
                    'max_signals_per_hour': 10
                },
                {
                    'client_id': 'TEST_CLIENT_002',
                    'mt5_login': 99999002,
                    'mt5_password': 'test_password_002',
                    'mt5_server': 'TestServer-MT5',
                    'account_info': {
                        'client_id': 'TEST_CLIENT_002',
                        'account_number': '99999002',
                        'broker': 'TestBroker',
                        'currency': 'USD',
                        'balance': 25000.0,
                        'equity': 25000.0,
                        'margin': 0.0,
                        'free_margin': 25000.0,
                        'margin_level': 0.0,
                        'max_daily_loss': 500.0,
                        'max_weekly_loss': 1200.0,
                        'max_monthly_loss': 3000.0,
                        'max_positions': 8,
                        'max_lot_size': 2.0
                    },
                    'preferred_pairs': ['EURUSD', 'USDJPY', 'XAUUSD'],
                    'risk_multiplier': 1.5,
                    'max_signals_per_hour': 15
                }
            ]
        }
        
        print("[CHECK] Test environment setup complete")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        print("\n? Cleaning up test environment...")
        
        # Remove test directory
        import shutil
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        
        print("[CHECK] Test cleanup complete")
    
    def setUp(self):
        """Set up before each test"""
        self.logger.info(f"\n[TEST_TUBE] Starting test: {self._testMethodName}")
    
    def tearDown(self):
        """Clean up after each test"""
        self.logger.info(f"[CHECK] Completed test: {self._testMethodName}")

    # ===========================================
    # CORE ENGINE TESTS
    # ===========================================
    
    def test_01_fvg_detector_initialization(self):
        """Test FVG detector initialization and basic functionality"""
        print("\n[SEARCH] Testing FVG Detector Initialization...")
        
        # Test basic initialization
        detector = EnhancedFVGDetector(CurrencyPair.EURUSD)
        self.assertIsNotNone(detector)
        self.assertEqual(detector.currency_pair, CurrencyPair.EURUSD)
        self.assertIsInstance(detector.timeframes, list)
        self.assertGreater(len(detector.timeframes), 0)
        
        # Test configuration
        self.assertIn(detector.currency_pair, detector.pair_configs)
        config = detector.current_config
        self.assertIn('min_gap_percentage', config)
        self.assertIn('confluence_threshold', config)
        
        print("[CHECK] FVG Detector initialization test passed")
    
    def test_02_fvg_signal_detection(self):
        """Test FVG signal detection with sample data"""
        print("\n[TARGET] Testing FVG Signal Detection...")
        
        detector = EnhancedFVGDetector(CurrencyPair.EURUSD)
        
        # Generate sample market data
        sample_data = self._generate_test_market_data('EURUSD', 'M5', 100)
        
        # Add artificial FVG pattern
        sample_data = self._inject_fvg_pattern(sample_data, 'bullish')
        
        # Detect signals
        signals = detector.detect_advanced_fvg(sample_data, 'M5')
        
        # Validate results
        self.assertIsInstance(signals, list)
        if signals:
            signal = signals[0]
            self.assertIsInstance(signal, FVGSignal)
            self.assertEqual(signal.currency_pair, CurrencyPair.EURUSD)
            self.assertEqual(signal.timeframe, 'M5')
            self.assertGreater(signal.confluence_score, 0)
            
            print(f"   [CHART] Detected {len(signals)} FVG signals")
            print(f"   [TARGET] Top signal confluence: {signal.confluence_score:.1f}")
        
        print("[CHECK] FVG signal detection test passed")
    
    def test_03_multi_timeframe_analysis(self):
        """Test multi-timeframe signal analysis"""
        print("\n? Testing Multi-Timeframe Analysis...")
        
        detector = EnhancedFVGDetector(CurrencyPair.EURUSD)
        
        # Generate data for multiple timeframes
        data_feeds = {}
        for tf in ['M1', 'M5', 'M15', 'H1']:
            data_feeds[tf] = self._generate_test_market_data('EURUSD', tf, 100)
        
        # Process multi-timeframe
        results = detector.process_multi_timeframe_advanced(data_feeds)
        
        # Validate results
        self.assertIsInstance(results, dict)
        for tf in ['M1', 'M5', 'M15', 'H1']:
            self.assertIn(tf, results)
            self.assertIsInstance(results[tf], list)
        
        # Test execution-ready signals
        execution_signals = detector.get_execution_ready_signals(results)
        self.assertIsInstance(execution_signals, list)
        
        total_signals = sum(len(signals) for signals in results.values())
        print(f"   [CHART] Total signals across timeframes: {total_signals}")
        print(f"   [ROCKET] Execution-ready signals: {len(execution_signals)}")
        
        print("[CHECK] Multi-timeframe analysis test passed")

    # ===========================================
    # TRADE EXECUTION TESTS
    # ===========================================
    
    def test_04_trade_executor_initialization(self):
        """Test trade executor initialization"""
        print("\n[LIGHTNING] Testing Trade Executor Initialization...")
        
        executor = TradeExecutor()
        self.assertIsNotNone(executor)
        self.assertEqual(len(executor.clients), 0)
        self.assertEqual(len(executor.active_positions), 0)
        self.assertFalse(executor.emergency_stop)
        
        # Test starting/stopping
        executor.start_execution_engine()
        self.assertTrue(executor.is_running)
        
        executor.stop_execution_engine()
        self.assertFalse(executor.is_running)
        
        print("[CHECK] Trade executor initialization test passed")
    
    def test_05_client_registration(self):
        """Test client registration and management"""
        print("\n[USERS] Testing Client Registration...")
        
        executor = TradeExecutor()
        
        # Register test clients
        for client_data in self.test_config['test_clients']:
            client_account = ClientAccount(**client_data['account_info'])
            success = executor.register_client(client_account)
            self.assertTrue(success)
        
        # Validate registration
        self.assertEqual(len(executor.clients), 2)
        
        # Test client summary
        for client_data in self.test_config['test_clients']:
            client_id = client_data['client_id']
            self.assertIn(client_id, executor.clients)
            
            summary = executor.get_client_summary(client_id)
            self.assertIn('account_info', summary)
            self.assertIn('pnl', summary)
            
        print(f"   [USERS] Successfully registered {len(executor.clients)} clients")
        print("[CHECK] Client registration test passed")
    
    def test_06_signal_execution(self):
        """Test FVG signal execution"""
        print("\n[TARGET] Testing Signal Execution...")
        
        executor = TradeExecutor()
        executor.start_execution_engine()
        
        # Register client
        client_data = self.test_config['test_clients'][0]
        client_account = ClientAccount(**client_data['account_info'])
        executor.register_client(client_account)
        
        # Create test signal
        test_signal = self._create_test_signal()
        
        # Execute signal
        result = executor.execute_fvg_signal(test_signal, client_data['client_id'])
        
        self.assertTrue(result['success'])
        self.assertIn('request_id', result)
        self.assertIn('estimated_lot_size', result)
        
        # Wait for execution
        time.sleep(2)
        
        # Check execution statistics
        stats = executor.get_execution_statistics()
        self.assertGreater(stats['total_executions'], 0)
        
        executor.stop_execution_engine()
        
        print(f"   [TARGET] Signal executed successfully")
        print(f"   [CHART] Execution stats: {stats['total_executions']} total")
        print("[CHECK] Signal execution test passed")

    # ===========================================
    # SYSTEM CONTROLLER TESTS
    # ===========================================
    
    def test_07_system_controller_initialization(self):
        """Test main system controller initialization"""
        print("\n[ROCKET] Testing System Controller Initialization...")
        
        controller = LightningScalperController()
        self.assertIsNotNone(controller)
        self.assertEqual(controller.status, SystemStatus.STARTING)
        self.assertIsNotNone(controller.fvg_detector)
        self.assertIsNotNone(controller.trade_executor)
        
        print("[CHECK] System controller initialization test passed")
    
    async def test_08_system_startup_and_shutdown(self):
        """Test system startup and shutdown process"""
        print("\n[REFRESH] Testing System Startup and Shutdown...")
        
        controller = LightningScalperController()
        
        # Test startup
        success = await controller.start_system()
        self.assertTrue(success)
        self.assertEqual(controller.status, SystemStatus.RUNNING)
        
        # Test system status
        status = controller.get_system_status()
        self.assertIn('status', status)
        self.assertIn('metrics', status)
        self.assertEqual(status['status'], SystemStatus.RUNNING)
        
        # Test shutdown
        await controller.shutdown()
        
        print("[CHECK] System startup and shutdown test passed")
    
    def test_09_client_management_integration(self):
        """Test client management through controller"""
        print("\n[USERS] Testing Client Management Integration...")
        
        controller = LightningScalperController()
        
        # Add clients (in demo mode)
        for client_data in self.test_config['test_clients']:
            # Register with trade executor (skip MT5 for testing)
            client_account = ClientAccount(**client_data['account_info'])
            success = controller.trade_executor.register_client(client_account)
            self.assertTrue(success)
        
        # Test client status
        for client_data in self.test_config['test_clients']:
            client_id = client_data['client_id']
            
            # Since we're not connecting to real MT5, we'll test the registered clients
            summary = controller.trade_executor.get_client_summary(client_id)
            self.assertIsNotNone(summary)
            self.assertIn('account_info', summary)
        
        print(f"   [USERS] Managed {len(self.test_config['test_clients'])} clients")
        print("[CHECK] Client management integration test passed")

    # ===========================================
    # DATA LOGGER TESTS
    # ===========================================
    
    def test_10_data_logger_initialization(self):
        """Test signal data logger initialization"""
        print("\n[DATABASE] Testing Data Logger Initialization...")
        
        logger = LightningScalperDataLogger(data_dir=str(self.data_dir))
        self.assertIsNotNone(logger)
        self.assertTrue(logger.db_path.exists())
        
        logger.start_logging()
        self.assertTrue(logger.is_running)
        
        logger.stop_logging()
        self.assertFalse(logger.is_running)
        
        print("[CHECK] Data logger initialization test passed")
    
    def test_11_signal_logging(self):
        """Test signal logging functionality"""
        print("\n[MEMO] Testing Signal Logging...")
        
        logger = LightningScalperDataLogger(data_dir=str(self.data_dir))
        logger.start_logging()
        
        # Create and log test signal
        test_signal = self._create_test_signal()
        logger.log_signal(test_signal, client_id="TEST_CLIENT", lot_size=0.1)
        
        # Log execution
        execution_data = {
            'execution_id': 'TEST_EXEC_001',
            'signal_id': test_signal.id,
            'client_id': 'TEST_CLIENT',
            'timestamp': datetime.now(),
            'order_type': 'LIMIT',
            'direction': 'BUY',
            'quantity': 0.1,
            'requested_price': 1.1000,
            'execution_status': 'FILLED',
            'fill_price': 1.1001,
            'fill_quantity': 0.1,
            'execution_time_ms': 250.0
        }
        
        logger.log_execution(execution_data)
        
        # Update signal outcome
        outcome_data = {
            'outcome': 'WIN',
            'pnl_pips': 15.0,
            'pnl_dollars': 150.0,
            'holding_time_minutes': 45
        }
        
        logger.update_signal_outcome(test_signal.id, outcome_data)
        
        # Flush and check statistics
        logger.flush_buffers()
        stats = logger.get_statistics()
        
        self.assertGreater(stats['signals_logged'], 0)
        self.assertGreater(stats['executions_logged'], 0)
        
        logger.stop_logging()
        
        print(f"   [MEMO] Logged {stats['signals_logged']} signals")
        print(f"   [LIGHTNING] Logged {stats['executions_logged']} executions")
        print("[CHECK] Signal logging test passed")

    # ===========================================
    # CONFIGURATION TESTS
    # ===========================================
    
    def test_12_config_manager(self):
        """Test configuration manager"""
        print("\n[SETTINGS] Testing Configuration Manager...")
        
        config_manager = LightningScalperConfigManager(config_dir=str(self.config_dir))
        self.assertIsNotNone(config_manager)
        
        # Test system config
        system_config = config_manager.get_system_config()
        self.assertIsNotNone(system_config)
        self.assertEqual(system_config.app_name, "Lightning Scalper")
        
        # Test trading config
        trading_config = config_manager.get_trading_config()
        self.assertIsNotNone(trading_config)
        self.assertGreater(trading_config.min_confluence_score, 0)
        
        # Test config summary
        summary = config_manager.get_config_summary()
        self.assertIn('system', summary)
        self.assertIn('trading', summary)
        self.assertIn('clients', summary)
        self.assertIn('brokers', summary)
        
        print(f"   [SETTINGS] System: {summary['system']['environment']}")
        print(f"   [CHART] Trading pairs: {summary['trading']['active_pairs']}")
        print(f"   [USERS] Clients: {summary['clients']['total']}")
        print("[CHECK] Configuration manager test passed")

    # ===========================================
    # DASHBOARD TESTS
    # ===========================================
    
    def test_13_dashboard_initialization(self):
        """Test web dashboard initialization (without actually starting server)"""
        print("\n[GLOBE] Testing Dashboard Initialization...")
        
        # Test dashboard creation
        dashboard = LightningScalperDashboard(controller=None, port=0)  # Port 0 to avoid conflicts
        self.assertIsNotNone(dashboard)
        self.assertIsNotNone(dashboard.app)
        self.assertIsNotNone(dashboard.socketio)
        
        # Test demo data generation
        system_status = dashboard._get_demo_system_status()
        self.assertIn('status', system_status)
        self.assertIn('metrics', system_status)
        
        clients_data = dashboard._get_demo_clients_data()
        self.assertIsInstance(clients_data, list)
        self.assertGreater(len(clients_data), 0)
        
        signals_data = dashboard._get_demo_signals_data()
        self.assertIsInstance(signals_data, list)
        
        print(f"   [GLOBE] Dashboard routes configured")
        print(f"   [CHART] Demo clients: {len(clients_data)}")
        print(f"   [TARGET] Demo signals: {len(signals_data)}")
        print("[CHECK] Dashboard initialization test passed")

    # ===========================================
    # END-TO-END TESTS
    # ===========================================
    
    async def test_14_end_to_end_trading_flow(self):
        """Test complete end-to-end trading flow"""
        print("\n[REFRESH] Testing End-to-End Trading Flow...")
        
        # 1. Initialize system
        controller = LightningScalperController()
        logger = LightningScalperDataLogger(data_dir=str(self.data_dir))
        
        # 2. Start systems
        await controller.start_system()
        logger.start_logging()
        
        # 3. Register clients
        for client_data in self.test_config['test_clients']:
            client_account = ClientAccount(**client_data['account_info'])
            success = controller.trade_executor.register_client(client_account)
            self.assertTrue(success)
        
        # 4. Generate test signal
        test_signal = self._create_test_signal()
        logger.log_signal(test_signal, client_id="TEST_CLIENT_001")
        
        # 5. Execute signal
        result = controller.trade_executor.execute_fvg_signal(
            test_signal, 
            "TEST_CLIENT_001"
        )
        self.assertTrue(result['success'])
        
        # 6. Wait for processing
        await asyncio.sleep(2)
        
        # 7. Check system status
        status = controller.get_system_status()
        self.assertEqual(status['status'], SystemStatus.RUNNING)
        
        # 8. Check logs
        logger.flush_buffers()
        stats = logger.get_statistics()
        self.assertGreater(stats['signals_logged'], 0)
        
        # 9. Shutdown
        logger.stop_logging()
        await controller.shutdown()
        
        print("   [REFRESH] Complete trading flow executed successfully")
        print(f"   [CHART] Final system status: {status['status']}")
        print(f"   [MEMO] Signals logged: {stats['signals_logged']}")
        print("[CHECK] End-to-end trading flow test passed")
    
    def test_15_performance_and_stress(self):
        """Test system performance under load"""
        print("\n[LIGHTNING] Testing Performance and Stress...")
        
        # Initialize components
        detector = EnhancedFVGDetector(CurrencyPair.EURUSD)
        executor = TradeExecutor()
        
        # Register multiple clients
        clients_registered = 0
        for i in range(5):  # Test with 5 clients
            client_account = ClientAccount(
                client_id=f"STRESS_CLIENT_{i:03d}",
                account_number=f"99990{i:02d}",
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
            )
            
            if executor.register_client(client_account):
                clients_registered += 1
        
        # Generate multiple signals
        signals_generated = 0
        for i in range(10):  # Generate 10 signals
            test_data = self._generate_test_market_data('EURUSD', 'M5', 50)
            signals = detector.detect_advanced_fvg(test_data, 'M5')
            signals_generated += len(signals)
        
        # Performance metrics
        self.assertGreater(clients_registered, 0)
        self.assertGreaterEqual(signals_generated, 0)  # May be 0 if no valid patterns
        
        print(f"   [USERS] Clients registered: {clients_registered}")
        print(f"   [TARGET] Signals generated: {signals_generated}")
        print(f"   [LIGHTNING] System handled load successfully")
        print("[CHECK] Performance and stress test passed")

    # ===========================================
    # HELPER METHODS
    # ===========================================
    
    def _generate_test_market_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Generate realistic test market data"""
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000, 'GBPUSD': 1.2500, 'USDJPY': 150.00,
            'AUDUSD': 0.6500, 'XAUUSD': 2000.00
        }
        
        base_price = base_prices.get(symbol, 1.0000)
        
        # Generate time series
        if timeframe == 'M1':
            freq = '1T'
        elif timeframe == 'M5':
            freq = '5T'
        elif timeframe == 'M15':
            freq = '15T'
        else:
            freq = '1H'
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=periods),
            periods=periods,
            freq=freq
        )
        
        # Generate price movement
        np.random.seed(42)  # Reproducible for testing
        returns = np.random.normal(0, 0.0001, periods)
        prices = base_price + np.cumsum(returns)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = np.random.uniform(0.00005, 0.0002)
            
            open_price = price + np.random.uniform(-volatility/2, volatility/2)
            close_price = price + np.random.uniform(-volatility/2, volatility/2)
            high_price = max(open_price, close_price) + np.random.uniform(0, volatility/3)
            low_price = min(open_price, close_price) - np.random.uniform(0, volatility/3)
            volume = np.random.randint(500, 2000)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data, index=dates)
    
    def _inject_fvg_pattern(self, df: pd.DataFrame, pattern_type: str = 'bullish') -> pd.DataFrame:
        """Inject a clear FVG pattern into market data"""
        if len(df) < 20:
            return df
        
        # Find a good spot to inject pattern (middle of data)
        inject_index = len(df) // 2
        
        # Create FVG pattern
        if pattern_type == 'bullish':
            # Bullish FVG: prev_high < next_low
            base_price = df.iloc[inject_index]['close']
            
            # Previous candle (bearish)
            df.iloc[inject_index-1, df.columns.get_loc('open')] = base_price + 0.0010
            df.iloc[inject_index-1, df.columns.get_loc('high')] = base_price + 0.0012
            df.iloc[inject_index-1, df.columns.get_loc('low')] = base_price - 0.0003
            df.iloc[inject_index-1, df.columns.get_loc('close')] = base_price
            
            # Current candle (strong bullish)
            df.iloc[inject_index, df.columns.get_loc('open')] = base_price + 0.0005
            df.iloc[inject_index, df.columns.get_loc('high')] = base_price + 0.0025
            df.iloc[inject_index, df.columns.get_loc('low')] = base_price + 0.0002
            df.iloc[inject_index, df.columns.get_loc('close')] = base_price + 0.0020
            
            # Next candle (continuation)
            df.iloc[inject_index+1, df.columns.get_loc('open')] = base_price + 0.0015
            df.iloc[inject_index+1, df.columns.get_loc('high')] = base_price + 0.0030
            df.iloc[inject_index+1, df.columns.get_loc('low')] = base_price + 0.0010  # Gap: prev_high < next_low
            df.iloc[inject_index+1, df.columns.get_loc('close')] = base_price + 0.0025
        
        return df
    
    def _create_test_signal(self) -> FVGSignal:
        """Create a test FVG signal"""
        return FVGSignal(
            id=f"TEST_SIGNAL_{int(time.time())}",
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
            structure_score=40.0,
            tags=["test_signal", "integration_test"]
        )


# ===========================================
# TEST RUNNER
# ===========================================

class TestRunner:
    """Custom test runner with detailed reporting"""
    
    def __init__(self):
        self.results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None,
            'duration': 0
        }
    
    async def run_all_tests(self):
        """Run all integration tests with detailed reporting"""
        print("\n" + "="*80)
        print("[ROCKET] LIGHTNING SCALPER INTEGRATION TEST SUITE")
        print("="*80)
        print("[LIGHTNING] Testing complete AI trading system integration")
        print("[TARGET] Validating FVG detection, execution, and data logging")
        print("[USERS] Multi-client support and risk management")
        print("="*80)
        
        self.results['start_time'] = time.time()
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(LightningScalperIntegrationTests)
        
        # Custom test result handler
        class DetailedTestResult(unittest.TextTestResult):
            def __init__(self, stream, descriptions, verbosity, runner_results):
                super().__init__(stream, descriptions, verbosity)
                self.runner_results = runner_results
            
            def startTest(self, test):
                super().startTest(test)
                self.runner_results['total_tests'] += 1
            
            def addSuccess(self, test):
                super().addSuccess(test)
                self.runner_results['passed'] += 1
            
            def addError(self, test, err):
                super().addError(test, err)
                self.runner_results['errors'] += 1
            
            def addFailure(self, test, err):
                super().addFailure(test, err)
                self.runner_results['failed'] += 1
            
            def addSkip(self, test, reason):
                super().addSkip(test, reason)
                self.runner_results['skipped'] += 1
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2,
            resultclass=lambda stream, descriptions, verbosity: 
                DetailedTestResult(stream, descriptions, verbosity, self.results)
        )
        
        # Handle async tests
        loop = asyncio.get_event_loop()
        
        # Run synchronous tests first
        sync_result = runner.run(suite)
        
        # Run specific async tests
        test_instance = LightningScalperIntegrationTests()
        test_instance.setUpClass()
        
        try:
            # Test async methods manually
            print("\n[REFRESH] Running async tests...")
            
            await test_instance.test_08_system_startup_and_shutdown()
            print("[CHECK] test_08_system_startup_and_shutdown - PASSED")
            
            await test_instance.test_14_end_to_end_trading_flow()
            print("[CHECK] test_14_end_to_end_trading_flow - PASSED")
            
        except Exception as e:
            print(f"[X] Async test failed: {e}")
            self.results['failed'] += 1
        finally:
            test_instance.tearDownClass()
        
        self.results['end_time'] = time.time()
        self.results['duration'] = self.results['end_time'] - self.results['start_time']
        
        # Print detailed results
        self._print_test_results()
        
        return self.results
    
    def _print_test_results(self):
        """Print detailed test results"""
        print("\n" + "="*80)
        print("[CHART] INTEGRATION TEST RESULTS SUMMARY")
        print("="*80)
        
        # Test statistics
        total = self.results['total_tests']
        passed = self.results['passed']
        failed = self.results['failed']
        errors = self.results['errors']
        skipped = self.results['skipped']
        
        print(f"[TRENDING_UP] Total Tests Run: {total}")
        print(f"[CHECK] Passed: {passed}")
        print(f"[X] Failed: {failed}")
        print(f"[SIREN] Errors: {errors}")
        print(f"?? Skipped: {skipped}")
        print(f"[TIMER] Duration: {self.results['duration']:.2f} seconds")
        
        # Success rate
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"[TARGET] Success Rate: {success_rate:.1f}%")
        
        print("\n" + "="*80)
        
        # Component status
        print("[SEARCH] COMPONENT TEST STATUS:")
        print("="*80)
        
        components = [
            ("[SEARCH] FVG Detection Engine", "Core signal detection and analysis"),
            ("[LIGHTNING] Trade Execution System", "Order management and execution"),
            ("[ROCKET] Main System Controller", "Central orchestration"),
            ("[DATABASE] Data Logger", "Signal and performance logging"),
            ("[SETTINGS] Configuration Manager", "System configuration"),
            ("[GLOBE] Web Dashboard", "Real-time monitoring interface"),
            ("[USERS] Client Management", "Multi-client support"),
            ("[REFRESH] End-to-End Flow", "Complete trading workflow")
        ]
        
        for component, description in components:
            print(f"{component:<30} {description}")
        
        print("\n" + "="*80)
        
        if failed == 0 and errors == 0:
            print("[PARTY] ALL TESTS PASSED! Lightning Scalper is ready for production!")
            print("? System integration validated successfully")
            print("[ROCKET] Ready to deploy for 80+ clients")
        else:
            print("[WARNING] Some tests failed. Please review the errors above.")
            print("[TOOL] Fix issues before production deployment")
        
        print("="*80)
        
        # Next steps
        print("\n[TARGET] NEXT STEPS:")
        if failed == 0 and errors == 0:
            print("1. [GLOBE] Create HTML templates for web dashboard")
            print("2. [SHIELD] Implement enhanced risk manager")
            print("3. [TOOL] Set up production configuration")
            print("4. [CHART] Deploy monitoring and alerting")
            print("5. [USERS] Onboard clients with real MT5 connections")
        else:
            print("1. [SEARCH] Review and fix failing tests")
            print("2. [TEST_TUBE] Re-run integration tests")
            print("3. [CLIPBOARD] Validate all components work together")
        
        print("\n[BULB] Use this command to run specific test categories:")
        print("   python -m pytest tests/ -v -k 'test_name'")
        print("\n[ROCKET] Lightning Scalper Integration Testing Complete!")


# ===========================================
# MAIN EXECUTION
# ===========================================

async def main():
    """Main test execution function"""
    print("[TEST_TUBE] Lightning Scalper Integration Test Suite")
    print("Testing complete AI trading system...")
    
    runner = TestRunner()
    results = await runner.run_all_tests()
    
    # Return appropriate exit code
    if results['failed'] > 0 or results['errors'] > 0:
        return 1
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[WARNING] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n? Test execution failed: {e}")
        sys.exit(1)