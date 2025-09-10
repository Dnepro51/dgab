import pandas as pd
import numpy as np
import sys
import os
sys.path.append('dgab')

from dgab.utils.validations import validate_inputs

# Test 1: Valid discrete data with integers
print("=== Test 1: Valid discrete data with integers ===")
try:
    df_int = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'launches': [1, 2, 3, 0, 1, 2]
    })
    validate_inputs(df_int, 'discrete', 'group', 'launches')
    print("✅ PASSED: Integer discrete data validation")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Valid discrete data with floats
print("\n=== Test 2: Valid discrete data with floats ===")
try:
    df_float = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'launches': [1.5, 2.3, 3.7, 0.0, 1.2, 2.8]
    })
    validate_inputs(df_float, 'discrete', 'group', 'launches')
    print("✅ PASSED: Float discrete data validation")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Valid discrete data with mixed int/float
print("\n=== Test 3: Valid discrete data with mixed int/float ===")
try:
    df_mixed = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'A', 'B'],
        'launches': [1, 2.5, 3, 0.0, 1.7, 2]
    })
    validate_inputs(df_mixed, 'discrete', 'group', 'launches')
    print("✅ PASSED: Mixed int/float discrete data validation")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: Invalid input - not DataFrame
print("\n=== Test 4: Invalid input - not DataFrame ===")
try:
    validate_inputs([1, 2, 3], 'discrete', 'group', 'launches')
    print("❌ FAILED: Should have raised TypeError")
except TypeError as e:
    print(f"✅ PASSED: Correctly caught TypeError - {e}")
except Exception as e:
    print(f"❌ FAILED: Wrong exception type - {e}")

# Test 5: Invalid input - missing column
print("\n=== Test 5: Invalid input - missing column ===")
try:
    df_missing = pd.DataFrame({
        'group': ['A', 'B'],
        'wrong_col': [1, 2]
    })
    validate_inputs(df_missing, 'discrete', 'group', 'launches')
    print("❌ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✅ PASSED: Correctly caught missing column - {e}")
except Exception as e:
    print(f"❌ FAILED: Wrong exception type - {e}")

# Test 6: Invalid input - string data in metric column
print("\n=== Test 6: Invalid input - string data in metric column ===")
try:
    df_string = pd.DataFrame({
        'group': ['A', 'B'],
        'launches': ['text', 'data']
    })
    validate_inputs(df_string, 'discrete', 'group', 'launches')
    print("❌ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✅ PASSED: Correctly caught non-numeric data - {e}")
except Exception as e:
    print(f"❌ FAILED: Wrong exception type - {e}")

# Test 7: Invalid input - insufficient groups
print("\n=== Test 7: Invalid input - insufficient groups ===")
try:
    df_one_group = pd.DataFrame({
        'group': ['A', 'A', 'A'],
        'launches': [1, 2, 3]
    })
    validate_inputs(df_one_group, 'discrete', 'group', 'launches')
    print("❌ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✅ PASSED: Correctly caught insufficient groups - {e}")
except Exception as e:
    print(f"❌ FAILED: Wrong exception type - {e}")

# Test 8: Invalid input - small sample size
print("\n=== Test 8: Invalid input - small sample size ===")
try:
    df_small = pd.DataFrame({
        'group': ['A', 'B'],
        'launches': [1, 2]
    })
    validate_inputs(df_small, 'discrete', 'group', 'launches')
    print("❌ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✅ PASSED: Correctly caught small sample size - {e}")
except Exception as e:
    print(f"❌ FAILED: Wrong exception type - {e}")

print("\n=== Testing with real data file ===")
try:
    df_real = pd.read_csv('discrete_2groups.csv')
    print(f"Data shape: {df_real.shape}")
    print(f"Columns: {df_real.columns.tolist()}")
    print(f"Data types: {df_real.dtypes.to_dict()}")
    print(f"Sample data:\n{df_real.head()}")
    
    validate_inputs(df_real, 'discrete', 'group', 'launches')
    print("✅ PASSED: Real discrete data validation")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n=== All tests completed ===")