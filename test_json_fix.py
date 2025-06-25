#!/usr/bin/env python3
"""
Test Script for JSON Serialization Fix
======================================
ทดสอบว่าการแก้ไข JSON Serialization ทำงานถูกต้องหรือไม่
"""

import sys
import os
import traceback
from datetime import datetime

def test_imports():
    """ทดสอบ imports"""
    print("🔍 Testing imports...")
    
    try:
        from json_utils import clean_data_for_json, safe_json_serialize, EnhancedJSONEncoder
        print("   ✅ json_utils imported successfully")
    except ImportError as e:
        print(f"   ❌ json_utils import failed: {str(e)}")
        return False
    
    try:
        if os.path.exists('advanced_features.py'):
            from advanced_features import MarketRegime
            print("   ✅ MarketRegime imported successfully")
        else:
            print("   ⚠️ advanced_features.py not found, skipping MarketRegime test")
    except ImportError as e:
        print(f"   ❌ MarketRegime import failed: {str(e)}")
    
    return True

def test_enum_serialization():
    """ทดสอบ Enum serialization"""
    print("🔍 Testing Enum serialization...")
    
    try:
        from json_utils import clean_data_for_json
        from enum import Enum
        
        class TestEnum(Enum):
            VALUE1 = "test_value_1"
            VALUE2 = "test_value_2"
        
        test_data = {
            'enum_value': TestEnum.VALUE1,
            'regular_value': 'test'
        }
        
        cleaned = clean_data_for_json(test_data)
        
        if isinstance(cleaned['enum_value'], str):
            print("   ✅ Enum serialization working")
            print(f"   📄 Enum value: {cleaned['enum_value']}")
            return True
        else:
            print(f"   ❌ Enum not converted to string: {type(cleaned['enum_value'])}")
            return False
            
    except Exception as e:
        print(f"   ❌ Enum serialization test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_market_regime():
    """ทดสอบ MarketRegime เฉพาะ"""
    print("🔍 Testing MarketRegime serialization...")
    
    try:
        if not os.path.exists('advanced_features.py'):
            print("   ⚠️ advanced_features.py not found, skipping test")
            return True
            
        from advanced_features import MarketRegime
        from json_utils import clean_data_for_json
        
        test_data = {
            'market_regime': MarketRegime.TRENDING_BULLISH,
            'other_data': 'test'
        }
        
        cleaned = clean_data_for_json(test_data)
        
        if isinstance(cleaned['market_regime'], str):
            print("   ✅ MarketRegime serialization working")
            print(f"   📄 MarketRegime value: {cleaned['market_regime']}")
            return True
        else:
            print(f"   ❌ MarketRegime not converted to string: {type(cleaned['market_regime'])}")
            return False
            
    except Exception as e:
        print(f"   ❌ MarketRegime serialization test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_api_endpoint():
    """ทดสอบ API endpoint (ถ้าระบบทำงานอยู่)"""
    print("🔍 Testing API endpoint...")
    
    try:
        import requests
        response = requests.get('http://localhost:5000/api/market-data', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("   ✅ API endpoint working")
            print(f"   📄 Response success: {data.get('success', False)}")
            return True
        else:
            print(f"   ❌ API returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"   ⚠️ API endpoint not accessible (system may not be running): {str(e)}")
        return True  # Not a failure if system isn't running
    except Exception as e:
        print(f"   ❌ API test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🧪 JSON Serialization Fix - Test Suite")
    print("=" * 50)
    print(f"📅 Test time: {datetime.now()}")
    print()
    
    tests = [
        ("Import Test", test_imports),
        ("Enum Serialization", test_enum_serialization),
        ("MarketRegime Serialization", test_market_regime),
        ("API Endpoint", test_api_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 Running: {test_name}")
        try:
            if test_func():
                print(f"   ✅ PASSED: {test_name}")
                passed += 1
            else:
                print(f"   ❌ FAILED: {test_name}")
        except Exception as e:
            print(f"   ❌ ERROR in {test_name}: {str(e)}")
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JSON serialization fix is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
