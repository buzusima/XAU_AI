"""
JSON Utilities for Smart Auto Trading System
===========================================
Utilities for handling JSON serialization of complex Python objects
including Enums, datetime, numpy arrays, pandas objects, etc.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Union

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Enhanced JSON Encoder for Smart Auto Trading System
    Handles MarketRegime enums, datetime objects, numpy arrays, etc.
    """
    
    def default(self, obj):
        try:
            # Handle Enum objects (including MarketRegime)
            if isinstance(obj, Enum):
                return obj.value
            
            # Handle datetime objects
            if isinstance(obj, datetime):
                return obj.isoformat()
            
            # Handle numpy types
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            
            # Handle pandas Series/DataFrame
            if isinstance(obj, pd.Series):
                return obj.to_dict()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            
            # Handle sets
            if isinstance(obj, set):
                return list(obj)
            
            # Handle complex numbers
            if isinstance(obj, complex):
                return {'real': obj.real, 'imag': obj.imag}
            
            # Handle any object with to_dict method
            if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
                return obj.to_dict()
            
            # Handle any object with __dict__
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            
            # Fallback to string representation
            return str(obj)
            
        except Exception as e:
            # Ultimate fallback - return type name
            return f"<Unserializable: {type(obj).__name__}>"

def safe_json_serialize(data: Any, indent: int = 2) -> str:
    """
    Safely serialize data to JSON string
    
    Args:
        data: Data to serialize
        indent: JSON indentation level
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(data, cls=EnhancedJSONEncoder, ensure_ascii=False, indent=indent)
    except Exception as e:
        print(f"⚠️ JSON serialization error: {str(e)}")
        # Create a safe fallback version
        safe_data = clean_data_for_json(data)
        return json.dumps(safe_data, ensure_ascii=False, indent=indent)

def clean_data_for_json(obj: Any) -> Any:
    """
    Recursively clean data for JSON serialization
    
    Args:
        obj: Object to clean
        
    Returns:
        JSON-serializable object
    """
    try:
        # Handle None
        if obj is None:
            return None
        
        # Handle basic JSON-serializable types
        if isinstance(obj, (str, int, float, bool)):
            return obj
        
        # Handle Enum objects
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [clean_data_for_json(item) for item in obj]
        
        # Handle dictionaries
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                # Ensure key is string
                clean_key = str(key) if not isinstance(key, str) else key
                cleaned[clean_key] = clean_data_for_json(value)
            return cleaned
        
        # Handle sets
        if isinstance(obj, set):
            return [clean_data_for_json(item) for item in obj]
        
        # Handle pandas objects
        if isinstance(obj, pd.Series):
            return clean_data_for_json(obj.to_dict())
        if isinstance(obj, pd.DataFrame):
            return clean_data_for_json(obj.to_dict('records'))
        
        # Handle complex numbers
        if isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        
        # Handle objects with to_dict method
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return clean_data_for_json(obj.to_dict())
        
        # Handle objects with __dict__
        if hasattr(obj, '__dict__'):
            return clean_data_for_json(obj.__dict__)
        
        # Fallback to string representation
        return str(obj)
        
    except Exception as e:
        print(f"⚠️ Error cleaning data: {str(e)}")
        return f"<Error: {str(e)}>"

def validate_json_serializable(obj: Any) -> bool:
    """
    Test if an object is JSON serializable
    
    Args:
        obj: Object to test
        
    Returns:
        True if serializable, False otherwise
    """
    try:
        json.dumps(obj, cls=EnhancedJSONEncoder)
        return True
    except (TypeError, ValueError):
        return False

def get_non_serializable_fields(obj: Dict) -> List[str]:
    """
    Get list of non-serializable fields in a dictionary
    
    Args:
        obj: Dictionary to check
        
    Returns:
        List of field names that are not JSON serializable
    """
    non_serializable = []
    
    if not isinstance(obj, dict):
        return non_serializable
    
    for key, value in obj.items():
        if not validate_json_serializable(value):
            non_serializable.append(key)
    
    return non_serializable

# Convenience functions
def safe_jsonify(data: Any) -> str:
    """Alias for safe_json_serialize"""
    return safe_json_serialize(data)

def clean_for_api(data: Any) -> Any:
    """Alias for clean_data_for_json"""
    return clean_data_for_json(data)

if __name__ == "__main__":
    # Test the utilities
    print("🧪 Testing JSON utilities...")
    
    # Test data
    from enum import Enum
    
    class TestEnum(Enum):
        VALUE1 = "test_value_1"
        VALUE2 = "test_value_2"
    
    test_data = {
        'string': 'test',
        'int': 42,
        'float': 3.14,
        'bool': True,
        'none': None,
        'list': [1, 2, 3],
        'dict': {'nested': 'value'},
        'enum': TestEnum.VALUE1,
        'datetime': datetime.now(),
        'set': {1, 2, 3},
        'complex': complex(1, 2)
    }
    
    print("✅ Original data:", test_data)
    cleaned = clean_data_for_json(test_data)
    print("✅ Cleaned data:", cleaned)
    
    json_str = safe_json_serialize(cleaned)
    print("✅ JSON string:", json_str[:100] + "..." if len(json_str) > 100 else json_str)
    
    print("🎯 JSON utilities test completed successfully!")
