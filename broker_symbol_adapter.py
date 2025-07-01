# 🔧 FIX 1: broker_symbol_adapter.py
# แก้ไขปัญหาการตรวจจับ symbol

import MetaTrader5 as mt5
import logging
from typing import Dict, List, Optional

class BrokerSymbolAdapter:
    """🔧 SYMBOL ADAPTER สำหรับรองรับหลายโบรกเกอร์ - FIXED"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 📊 ระบบเดิมใช้ .c (IC Markets style)
        self.system_symbols = [
            'EURUSD.c', 'GBPUSD.c', 'USDJPY.c', 'USDCHF.c', 'AUDUSD.c', 'NZDUSD.c', 'USDCAD.c',
            'EURGBP.c', 'EURJPY.c', 'EURCHF.c', 'EURAUD.c', 'EURNZD.c', 'EURCAD.c',
            'GBPJPY.c', 'GBPCHF.c', 'GBPAUD.c', 'GBPNZD.c', 'GBPCAD.c',
            'AUDCHF.c', 'AUDJPY.c', 'AUDNZD.c', 'AUDCAD.c',
            'NZDJPY.c', 'NZDCHF.c', 'NZDCAD.c',
            'CHFJPY.c', 'CADJPY.c', 'XAUUSD.c'
        ]
        
        # Runtime Detection
        self.detected_broker = None
        self.broker_symbol_map = {}
        self.reverse_symbol_map = {}
        self.mapping_successful = False
        
    def detect_and_map_broker(self) -> bool:
        """🎯 AUTO-DETECT และสร้าง symbol mapping อัตโนมัติ - ENHANCED"""
        try:
            if not mt5.initialize():
                self.logger.error("MT5 not connected")
                return False
            
            # แสดงข้อมูล Account
            account_info = mt5.account_info()
            if account_info and account_info.server:
                server_name = account_info.server
                self.logger.info(f"🏦 Connected to: {server_name}")
                self.logger.info(f"💰 Account: {account_info.login}")
            
            # 🎯 ตรวจจับ suffix แบบอัตโนมัติ
            success = self._create_symbol_mapping()
            
            if success:
                self.mapping_successful = True
                self.logger.info("✅ Symbol mapping completed successfully")
            else:
                self.logger.warning("⚠️ Symbol mapping failed, using fallback")
                self._create_fallback_mapping()
                
            return success
                
        except Exception as e:
            self.logger.error(f"Auto-detection failed: {str(e)}")
            self._create_fallback_mapping()
            return False
    
    def _create_symbol_mapping(self) -> bool:
        """🎯 สร้าง mapping โดยตรวจจับ suffix อัตโนมัติ - FIXED"""
        try:
            # ดึง symbols ทั้งหมดจาก MT5
            available_symbols = mt5.symbols_get()
            if not available_symbols:
                self.logger.error("No symbols available from MT5")
                return False
            
            available_symbol_names = [s.name for s in available_symbols]
            self.logger.info(f"🔍 Found {len(available_symbol_names)} symbols in MT5")
            
            # Clear previous mappings
            self.broker_symbol_map = {}
            self.reverse_symbol_map = {}
            
            # 🎯 AUTO-DETECT SUFFIX PATTERNS
            detected_suffixes = self._detect_symbol_suffixes(available_symbol_names)
            self.logger.info(f"🔧 Detected suffixes: {detected_suffixes}")
            
            mapped_count = 0
            for system_symbol in self.system_symbols:
                # ลบ .c ออกเพื่อหา base symbol
                base_symbol = system_symbol.replace('.c', '')
                
                # ลองหา symbol ที่ตรงกับ base + detected suffixes
                broker_symbol = self._find_matching_symbol(base_symbol, available_symbol_names, detected_suffixes)
                
                if broker_symbol:
                    # สร้าง mapping
                    self.broker_symbol_map[system_symbol] = broker_symbol
                    self.reverse_symbol_map[broker_symbol] = system_symbol
                    mapped_count += 1
                    
                    # Make symbol visible in MT5
                    try:
                        symbol_info = mt5.symbol_info(broker_symbol)
                        if symbol_info and not symbol_info.visible:
                            mt5.symbol_select(broker_symbol, True)
                    except Exception as e:
                        self.logger.warning(f"Could not make {broker_symbol} visible: {str(e)}")
                else:
                    self.logger.warning(f"⚠️ Could not find broker symbol for {system_symbol}")
            
            self.logger.info(f"✅ Successfully mapped {mapped_count}/{len(self.system_symbols)} symbols")
            
            # แสดง mapping ที่สร้างได้
            if mapped_count > 0:
                self.logger.info("📊 Symbol Mapping Created:")
                for sys_sym, broker_sym in list(self.broker_symbol_map.items())[:5]:
                    self.logger.info(f"   {sys_sym} → {broker_sym}")
                if len(self.broker_symbol_map) > 5:
                    self.logger.info(f"   ... และอีก {len(self.broker_symbol_map)-5} symbols")
            
            return mapped_count > 0
            
        except Exception as e:
            self.logger.error(f"Symbol mapping creation failed: {str(e)}")
            return False
    
    def _detect_symbol_suffixes(self, available_symbols: List[str]) -> List[str]:
        """🔍 ตรวจจับ suffix ทั้งหมดที่มีใน MT5 - ENHANCED"""
        try:
            # Base symbols ที่ต้องการตรวจสอบ
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
            
            detected_suffixes = set()
            
            for symbol_name in available_symbols:
                for base_pair in major_pairs:
                    if symbol_name.startswith(base_pair) and len(symbol_name) <= len(base_pair) + 10:
                        suffix = symbol_name[len(base_pair):]
                        if len(suffix) <= 5:  # reasonable suffix length
                            detected_suffixes.add(suffix if suffix else 'none')
            
            # เรียงลำดับตามความนิยม
            suffix_list = sorted(list(detected_suffixes))
            
            # ใส่ suffix ที่พบบ่อยไว้ข้างหน้า
            common_suffixes = ['none', '.c', '.raw', '.ecn', '.pro', '.m']
            ordered_suffixes = []
            
            for common in common_suffixes:
                if common in suffix_list:
                    ordered_suffixes.append(common)
                    suffix_list.remove(common)
            
            # เพิ่มที่เหลือ
            ordered_suffixes.extend(suffix_list)
            
            return ordered_suffixes
            
        except Exception as e:
            self.logger.error(f"Suffix detection error: {str(e)}")
            return ['none', '.c', '.raw', '.ecn']  # fallback
    
    def _find_matching_symbol(self, base_symbol: str, available_symbol_names: List[str], detected_suffixes: List[str]) -> Optional[str]:
        """🔍 หา symbol ที่ตรงกับ base + suffix - COMPLETELY FIXED"""
        try:
            # ลองหาด้วย detected suffixes
            for suffix in detected_suffixes:
                if suffix == 'none':
                    test_symbol = base_symbol
                else:
                    test_symbol = base_symbol + suffix
                
                if test_symbol in available_symbol_names:
                    # ทดสอบว่า symbol ใช้งานได้จริง
                    try:
                        symbol_info = mt5.symbol_info(test_symbol)
                        if symbol_info:
                            self.logger.debug(f"✅ Found: {base_symbol} → {test_symbol}")
                            return test_symbol
                    except Exception:
                        continue
            
            # ถ้าหาไม่เจอ ลองใช้ fuzzy matching
            for symbol_name in available_symbol_names:
                if (symbol_name.startswith(base_symbol) and 
                    len(symbol_name) <= len(base_symbol) + 10 and
                    len(symbol_name) >= len(base_symbol)):
                    
                    # ทดสอบว่า symbol ใช้งานได้จริง
                    try:
                        symbol_info = mt5.symbol_info(symbol_name)
                        if symbol_info:
                            self.logger.debug(f"✅ Fuzzy match: {base_symbol} → {symbol_name}")
                            return symbol_name
                    except Exception:
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol matching failed for {base_symbol}: {str(e)}")
            return None
    
    def _create_fallback_mapping(self):
        """สร้าง fallback mapping 1:1"""
        self.logger.info("🔄 Creating fallback 1:1 mapping")
        for system_symbol in self.system_symbols:
            self.broker_symbol_map[system_symbol] = system_symbol
            self.reverse_symbol_map[system_symbol] = system_symbol
        self.mapping_successful = False
    
    def system_to_broker_symbol(self, system_symbol: str) -> str:
        """🔄 แปลง system symbol เป็น broker symbol - ENHANCED"""
        try:
            broker_symbol = self.broker_symbol_map.get(system_symbol, system_symbol)
            
            # Validate broker symbol exists in MT5 (ถ้าไม่ใช่ fallback mode)
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
        """📋 ดึงข้อมูล mapping ทั้งหมด - ENHANCED"""
        try:
            account_info = mt5.account_info()
            server_name = account_info.server if account_info else "Unknown"
            
            return {
                'server': server_name,
                'mapping_successful': self.mapping_successful,
                'total_system_symbols': len(self.system_symbols),
                'mapped_symbols': len(self.broker_symbol_map),
                'mapping_success_rate': f"{len(self.broker_symbol_map)/len(self.system_symbols)*100:.1f}%",
                'available_broker_symbols': list(self.broker_symbol_map.values()),
                'sample_mapping': dict(list(self.broker_symbol_map.items())[:5]),
                'detected_suffixes': self._get_unique_suffixes(),
                'status': 'ACTIVE' if self.mapping_successful else 'FALLBACK'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'ERROR'}
    
    def _get_unique_suffixes(self) -> List[str]:
        """หา suffix ที่ใช้อยู่"""
        suffixes = set()
        for broker_symbol in self.broker_symbol_map.values():
            for system_symbol in self.system_symbols:
                base = system_symbol.replace('.c', '')
                if broker_symbol.startswith(base):
                    suffix = broker_symbol[len(base):]
                    suffixes.add(suffix if suffix else 'none')
                    break
        return sorted(list(suffixes))

    def test_symbol_mapping(self) -> Dict:
        """🧪 ทดสอบ symbol mapping"""
        test_results = []
        
        for system_symbol in self.system_symbols[:5]:  # Test first 5
            broker_symbol = self.system_to_broker_symbol(system_symbol)
            
            try:
                symbol_info = mt5.symbol_info(broker_symbol)
                tick = mt5.symbol_info_tick(broker_symbol)
                
                test_results.append({
                    'system_symbol': system_symbol,
                    'broker_symbol': broker_symbol,
                    'symbol_exists': symbol_info is not None,
                    'has_price': tick is not None and tick.bid > 0,
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
            'success_rate': f"{success_count/len(test_results)*100:.1f}%",
            'mapping_working': success_count > 0
        }

print("🔧 BROKER SYMBOL ADAPTER - FIXED!")
print("✅ Enhanced suffix detection")
print("✅ Improved symbol matching")  
print("✅ Better error handling")
print("✅ Fallback mechanisms")
print("✅ Symbol validation")