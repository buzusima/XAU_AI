# 🔧 FIX 1: broker_symbol_adapter.py
# แก้ไขปัญหาการตรวจจับ symbol

import MetaTrader5 as mt5
import logging
from typing import Dict, List, Optional

class BrokerSymbolAdapter:
    """🔧 ENHANCED SYMBOL ADAPTER - FIX KeyError"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.system_symbols = []

        # Base symbols (ไม่มี suffix)
        self.base_symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD',
            'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURNZD', 'EURCAD',
            'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD',
            'AUDCHF', 'AUDJPY', 'AUDNZD', 'AUDCAD',
            'NZDJPY', 'NZDCHF', 'NZDCAD',
            'CHFJPY', 'CADJPY', 'XAUUSD'
        ]
        self.system_symbols = [base + '.c' for base in self.base_symbols]
        # Runtime Detection Results
        self.detected_suffix = '.v'  # Default based on current broker
        self.detected_suffixes = ['.v', '.c']
        self.server_name = "Unknown"
        self.broker_symbol_map = {}  # system_symbol -> broker_symbol
        self.reverse_symbol_map = {}  # broker_symbol -> system_symbol
        self.mapping_successful = False
        
    def detect_and_map_broker(self) -> bool:
        """🎯 Auto-detect broker symbols และสร้าง mapping"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 not connected")
                return False
            
            # Get account info และเก็บ server name
            account_info = mt5.account_info()
            if account_info and account_info.server:
                self.server_name = account_info.server
                self.logger.info(f"🏦 Connected to: {self.server_name}")
            else:
                self.server_name = "Unknown Server"
                self.logger.warning("⚠️ Cannot get server name")
            
            # Get all available symbols
            all_mt5_symbols = mt5.symbols_get()
            if not all_mt5_symbols:
                self.logger.error("❌ No symbols available from MT5")
                return False
            
            self.logger.info(f"📊 Total MT5 symbols available: {len(all_mt5_symbols)}")
            
            # Find our forex symbols with their actual suffixes
            detected_symbols = {}
            suffix_count = {}
            
            for mt5_symbol in all_mt5_symbols:
                symbol_name = mt5_symbol.name
                
                # Check if this symbol matches any of our base symbols
                for base_symbol in self.base_symbols:
                    if symbol_name.startswith(base_symbol):
                        # Extract suffix
                        suffix = symbol_name[len(base_symbol):]
                        
                        # Only consider tradeable symbols
                        if mt5_symbol.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                            # เก็บเฉพาะ suffix ที่ดีที่สุดสำหรับแต่ละ base symbol
                            if base_symbol not in detected_symbols:
                                detected_symbols[base_symbol] = symbol_name
                                suffix_count[suffix] = suffix_count.get(suffix, 0) + 1
                                self.logger.debug(f"✅ Found: {base_symbol} → {symbol_name}")
                            break
            
            # Determine the most common suffix
            if suffix_count:
                self.detected_suffix = max(suffix_count, key=suffix_count.get)
                self.detected_suffixes = list(suffix_count.keys())
                self.logger.info(f"🔧 Detected primary suffix: '{self.detected_suffix}' (used by {suffix_count[self.detected_suffix]} symbols)")
                self.logger.info(f"📋 All suffixes found: {dict(suffix_count)}")
            else:
                self.logger.error("❌ No forex symbols detected")
                return False
            
            # Build the mapping
            for base_symbol in self.base_symbols:
                system_symbol = base_symbol + '.c'  # System uses .c
                broker_symbol = detected_symbols.get(base_symbol)
                
                if broker_symbol:
                    self.broker_symbol_map[system_symbol] = broker_symbol
                    self.reverse_symbol_map[broker_symbol] = system_symbol
            
            # Log results
            self.logger.info(f"✅ Symbol mapping completed!")
            self.logger.info(f"📊 Mapped symbols: {len(self.broker_symbol_map)}/{len(self.system_symbols)}")
            self.logger.info(f"🎯 Success rate: {len(self.broker_symbol_map)/len(self.system_symbols)*100:.1f}%")
            
            # Show sample mapping
            sample_count = min(5, len(self.broker_symbol_map))
            sample_items = list(self.broker_symbol_map.items())[:sample_count]
            self.logger.info("📋 Sample mappings:")
            for system, broker in sample_items:
                self.logger.info(f"   {system} → {broker}")
            
            self.mapping_successful = len(self.broker_symbol_map) >= len(self.system_symbols) * 0.5
            
            if self.mapping_successful:
                self.logger.info("🎉 Symbol mapping successful!")
                return True
            else:
                self.logger.warning("⚠️ Symbol mapping had low success rate")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in broker detection: {str(e)}")
            return False
    
    def system_to_broker_symbol(self, system_symbol: str) -> str:
        """🔄 แปลง system symbol เป็น broker symbol"""
        try:
            broker_symbol = self.broker_symbol_map.get(system_symbol, system_symbol)
            
            # Validate broker symbol exists in MT5
            if self.mapping_successful and broker_symbol != system_symbol:
                try:
                    symbol_info = mt5.symbol_info(broker_symbol)
                    if symbol_info is None:
                        self.logger.warning(f"⚠️ Broker symbol {broker_symbol} not found, using {system_symbol}")
                        return system_symbol
                except Exception:
                    pass
                    
            return broker_symbol
        except Exception as e:
            self.logger.error(f"Error converting {system_symbol}: {str(e)}")
            return system_symbol
    
    def broker_to_system_symbol(self, broker_symbol: str) -> str:
        """🔄 แปลง broker symbol เป็น system symbol"""
        return self.reverse_symbol_map.get(broker_symbol, broker_symbol)
    
    def get_mapped_symbols(self) -> List[str]:
        """📊 ดึงรายการ broker symbols ที่ map ได้แล้ว"""
        if not self.broker_symbol_map:
            return self.system_symbols  # fallback
        return list(self.broker_symbol_map.values())
    
    def get_mapping_info(self) -> Dict:
        """📊 ดึงข้อมูล mapping ทั้งหมด - COMPLETE VERSION"""
        try:
            # Get server name from MT5 if not available
            if self.server_name == "Unknown":
                account_info = mt5.account_info()
                self.server_name = account_info.server if account_info else "Unknown"
            
            if len(self.system_symbols) > 0:
                mapping_success_rate = f"{len(self.broker_symbol_map)/len(self.system_symbols)*100:.1f}%"
            else:
                mapping_success_rate = "0%"

            return {
                'server': self.server_name,
                'detected_suffix': self.detected_suffix,
                'detected_suffixes': self.detected_suffixes,
                'total_system_symbols': len(self.system_symbols),
                'mapped_symbols': len(self.broker_symbol_map),
                'mapping_success_rate': mapping_success_rate,
                'broker_symbols': list(self.broker_symbol_map.values()),
                'system_symbols': self.system_symbols,
                'symbol_mapping': self.broker_symbol_map,
                'sample_mapping': dict(list(self.broker_symbol_map.items())[:5]),
                'mapping_successful': self.mapping_successful,
                'available_broker_symbols': list(self.broker_symbol_map.values())
            }
        except Exception as e:
            self.logger.error(f"Error getting mapping info: {str(e)}")
            return {
                'server': self.server_name,
                'detected_suffix': self.detected_suffix,
                'detected_suffixes': self.detected_suffixes,
                'total_system_symbols': len(self.system_symbols),
                'mapped_symbols': 0,
                'mapping_success_rate': '0%',
                'broker_symbols': [],
                'system_symbols': self.system_symbols,
                'symbol_mapping': {},
                'sample_mapping': {},
                'mapping_successful': False,
                'available_broker_symbols': [],
                'error': str(e)
            }
    
    def test_symbol_mapping(self) -> Dict:
        """🧪 ทดสอบ symbol mapping"""
        test_results = []
        
        test_symbols = list(self.system_symbols)[:5]  # Test first 5
        
        for system_symbol in test_symbols:
            broker_symbol = self.system_to_broker_symbol(system_symbol)
            
            try:
                symbol_info = mt5.symbol_info(broker_symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                
                test_results.append({
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol,
                    'symbol_exists': symbol_info is not None,
                    'has_price': tick is not None and tick.bid > 0 if tick else False,
                    'status': 'OK' if symbol_info and tick and tick.bid > 0 else 'FAILED'
                })
            except Exception as e:
                test_results.append({
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        success_count = len([r for r in test_results if r.get('status') == 'OK'])
        
        return {
            'test_results': test_results,
            'success_count': success_count,
            'total_tested': len(test_results),
            'success_rate': f"{success_count/len(test_results)*100:.1f}%" if test_results else "0%",
            'mapping_working': success_count > 0
        }
    
    def get_available_symbols(self) -> List[str]:
        """📋 ดึงรายการ symbols ที่ใช้ได้"""
        if self.mapping_successful:
            return list(self.broker_symbol_map.values())
        else:
            return self.system_symbols
    
    def is_symbol_available(self, symbol: str) -> bool:
        """✅ เช็คว่า symbol ใช้ได้หรือไม่"""
        try:
            symbol_info = mt5.symbol_info(symbol)
            return symbol_info is not None and symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL
        except Exception:
            return False
    
    def refresh_mapping(self) -> bool:
        """🔄 Refresh symbol mapping"""
        self.logger.info("🔄 Refreshing symbol mapping...")
        self.broker_symbol_map.clear()
        self.reverse_symbol_map.clear()
        return self.detect_and_map_broker()