# 🎯 BROKER SYMBOL ADAPTER - เพิ่มเข้าระบบเดิม
# แค่เพิ่มไฟล์นี้เข้าไปใน project เดิม ไม่ต้องแก้อะไรเลย!

import MetaTrader5 as mt5
import logging
from typing import Dict, List, Optional

class BrokerSymbolAdapter:
    """
    🔧 SYMBOL ADAPTER สำหรับรองรับหลายโบรกเกอร์
    ========================================
    เพิ่มเข้าระบบเดิมที่ใช้ .c อยู่แล้ว ไม่ต้องแก้โค้ดเดิม
    """
    
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
        
    def detect_and_map_broker(self) -> bool:
        """
        🎯 AUTO-DETECT และสร้าง symbol mapping อัตโนมัติ
        ไม่ต้องรู้ชื่อโบรกเกอร์ ตรวจจับ suffix เอง!
        """
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
            
            # 🎯 ตรวจจับ suffix แบบอัตโนมัติ (ไม่ต้องรู้ชื่อโบรกเกอร์)
            return self._create_symbol_mapping()
                
        except Exception as e:
            self.logger.error(f"Auto-detection failed: {str(e)}")
            return False
    
    def _create_symbol_mapping(self) -> bool:
        """🎯 สร้าง mapping โดยตรวจจับ suffix อัตโนมัติ"""
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
                    symbol_info = mt5.symbol_info(broker_symbol)
                    if symbol_info and not symbol_info.visible:
                        mt5.symbol_select(broker_symbol, True)
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
        """🔍 ตรวจจับ suffix ทั้งหมดที่มีใน MT5"""
        try:
            # Base symbols ที่ต้องการตรวจสอบ
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
            
            detected_suffixes = set()
            
            for symbol_name in available_symbols:
                for base_pair in major_pairs:
                    if symbol_name.startswith(base_pair):
                        # หา suffix
                        suffix = symbol_name[len(base_pair):]
                        if suffix:  # มี suffix
                            detected_suffixes.add(suffix)
                        else:  # ไม่มี suffix
                            detected_suffixes.add('')
                        break
            
            # เรียงลำดับตาม priority (ไม่มี suffix = สูงสุด, .c = รอง)
            suffix_priority = ['', '.c', '.raw', '.ecn', '.stp', '.pro', '.m']
            sorted_suffixes = []
            
            for priority_suffix in suffix_priority:
                if priority_suffix in detected_suffixes:
                    sorted_suffixes.append(priority_suffix)
            
            # เพิ่ม suffix อื่นๆ ที่เหลือ
            for suffix in detected_suffixes:
                if suffix not in sorted_suffixes:
                    sorted_suffixes.append(suffix)
            
            return sorted_suffixes
            
        except Exception as e:
            self.logger.error(f"Suffix detection failed: {str(e)}")
            return ['', '.c']  # fallback
    
    def _find_matching_symbol(self, base_symbol: str, available_symbols: List[str], suffixes: List[str]) -> Optional[str]:
        """🎯 หา broker symbol ที่ตรงกับ base symbol"""
        try:
            # ลองแต่ละ suffix ตามลำดับ priority
            for suffix in suffixes:
                candidate = base_symbol + suffix
                if candidate in available_symbols:
                    return candidate
            
            # ถ้าไม่เจอ ลองหาแบบ case-insensitive
            base_lower = base_symbol.lower()
            for symbol in available_symbols:
                if symbol.lower().startswith(base_lower):
                    # ตรวจสอบว่าเป็น exact match หรือมี suffix เท่านั้น
                    remaining = symbol[len(base_symbol):]
                    if not remaining or remaining.startswith('.') or remaining.startswith('_'):
                        return symbol
            
            return None
            
        except Exception as e:
            self.logger.error(f"Symbol matching failed: {str(e)}")
            return None
    
    def system_to_broker_symbol(self, system_symbol: str) -> str:
        """🔄 แปลง system symbol เป็น broker symbol"""
        return self.broker_symbol_map.get(system_symbol, system_symbol)
    
    def broker_to_system_symbol(self, broker_symbol: str) -> str:
        """🔄 แปลง broker symbol เป็น system symbol"""
        return self.reverse_symbol_map.get(broker_symbol, broker_symbol)
    
    def get_mapped_symbols(self) -> List[str]:
        """📊 ดึงรายการ broker symbols ที่ map ได้แล้ว"""
        return list(self.broker_symbol_map.values())
    
    def get_mapping_info(self) -> Dict:
        """📋 ดึงข้อมูล mapping ทั้งหมด"""
        try:
            account_info = mt5.account_info()
            server_name = account_info.server if account_info else "Unknown"
            
            return {
                'server': server_name,
                'total_system_symbols': len(self.system_symbols),
                'mapped_symbols': len(self.broker_symbol_map),
                'mapping_success_rate': f"{len(self.broker_symbol_map)/len(self.system_symbols)*100:.1f}%",
                'available_broker_symbols': list(self.broker_symbol_map.values()),
                'sample_mapping': dict(list(self.broker_symbol_map.items())[:5]),
                'detected_suffixes': self._get_unique_suffixes()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _get_unique_suffixes(self) -> List[str]:
        """หา suffix ที่ใช้อยู่"""
        suffixes = set()
        for broker_symbol in self.broker_symbol_map.values():
            # หา base symbol
            for system_symbol in self.system_symbols:
                base = system_symbol.replace('.c', '')
                if broker_symbol.startswith(base):
                    suffix = broker_symbol[len(base):]
                    suffixes.add(suffix if suffix else 'none')
                    break
        return sorted(list(suffixes))

# 🎯 การใช้งาน - เพิ่มเข้าระบบเดิม
"""
# 1. เพิ่มใน __init__ ของ main trading system
def __init__(self):
    # ... existing code ...
    
    # เพิ่ม Symbol Adapter
    self.symbol_adapter = BrokerSymbolAdapter()
    if self.symbol_adapter.detect_and_map_broker():
        print("✅ Broker symbols mapped successfully!")
        print(f"📊 Mapping info: {self.symbol_adapter.get_mapping_info()}")
    else:
        print("⚠️ Using default symbol format")

# 2. ใช้ในส่วนที่เรียก MT5 functions
def get_symbol_data(self, system_symbol):
    # แปลง system symbol เป็น broker symbol
    broker_symbol = self.symbol_adapter.system_to_broker_symbol(system_symbol)
    
    # เรียก MT5 ด้วย broker symbol
    rates = mt5.copy_rates_from_pos(broker_symbol, mt5.TIMEFRAME_H1, 0, 100)
    return rates

# 3. ใช้ในส่วน signal generation
def generate_signals(self):
    signals = {}
    
    # ใช้ mapped symbols แทน hardcoded .c
    for broker_symbol in self.symbol_adapter.get_mapped_symbols():
        system_symbol = self.symbol_adapter.broker_to_system_symbol(broker_symbol)
        
        # Generate signal using broker_symbol for MT5 calls
        signal = self.calculate_signal(broker_symbol)
        
        # Store result using system_symbol for consistency
        signals[system_symbol] = signal
    
    return signals
"""

print("🎯 SMART SYMBOL ADAPTER - AUTO-DETECTION")
print("✅ ตรวจจับ suffix อัตโนมัติ")
print("🔧 ไม่ต้องรู้ชื่อโบรกเกอร์")
print("📊 รองรับทุกรูปแบบ symbol")
print("🚀 เพิ่มเข้าระบบเดิมได้ทันที!")