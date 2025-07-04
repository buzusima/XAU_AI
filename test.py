# test_import.py
print("Testing pullback_protection import...")

try:
    print("1. Importing PullbackProtectionPlugin...")
    from pullback_protection import PullbackProtectionPlugin
    print("   ✅ SUCCESS")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

try:
    print("2. Importing integrate_with_main_system...")
    from pullback_protection import integrate_with_main_system
    print("   ✅ SUCCESS")
    print(f"   Function type: {type(integrate_with_main_system)}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# ทดสอบ syntax
try:
    print("3. Testing file syntax...")
    import pullback_protection
    print("   ✅ Module loads successfully")
    
    # ดู functions ทั้งหมดในไฟล์
    functions = [name for name in dir(pullback_protection) if not name.startswith('_')]
    print(f"   Available functions: {functions}")
    
except Exception as e:
    print(f"   ❌ Syntax error: {e}")