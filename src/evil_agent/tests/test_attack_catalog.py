"""
VecSec Attack Catalog Functional Diagnostic
Tests real runtime behavior of attack_catalog.py subsystems
Purpose: Diagnose all attack catalog issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Attack Catalog Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables"""
    pass  # Attack catalog doesn't use env vars


# ============================================================================
# 1️⃣ Attack Types Test
# ============================================================================

def test_attack_types():
    """Test that attack types are defined"""
    print("\n🔧 Testing Attack Types...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import ATTACK_TYPES, get_attack_types
        
        # Test ATTACK_TYPES structure
        print(f"   ATTACK_TYPES keys: {list(ATTACK_TYPES.keys())}")
        
        expected_types = [
            "prompt_injection",
            "data_exfiltration",
            "social_engineering",
            "obfuscation",
            "jailbreak",
            "poisoning",
            "privilege_escalation"
        ]
        
        for attack_type in expected_types:
            assert attack_type in ATTACK_TYPES, f"Should have {attack_type}"
            print(f"   ✅ {attack_type}: {len(ATTACK_TYPES[attack_type])} attacks")
        
        # Test get_attack_types function
        all_types = get_attack_types()
        assert all_types == ATTACK_TYPES, "get_attack_types should return ATTACK_TYPES"
        print(f"   ✅ get_attack_types() works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Hardcoded Attacks Test (CRITICAL)
# ============================================================================

def test_hardcoded_attacks():
    """Test that attacks are hardcoded (not configurable)"""
    print("\n🔴 Testing Hardcoded Attacks...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test attack structure
        print(f"   Attack types: {len(ATTACK_TYPES)} types")
        total_attacks = sum(len(attacks) for attacks in ATTACK_TYPES.values())
        print(f"   Total attacks: {total_attacks}")
        
        for attack_type, attacks in ATTACK_TYPES.items():
            print(f"   {attack_type}: {len(attacks)} attacks")
            if len(attacks) > 0:
                print(f"     Example: {attacks[0][:60]}...")
        
        print(f"   ⚠️  ISSUE: Attacks are hardcoded (not configurable)")
        print(f"   ⚠️  Expected: Configurable via file/DB/API")
        print(f"   ⚠️  Actual: Hardcoded in Python code")
        print(f"   ⚠️  Impact: Cannot add new attacks without code changes")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Get Attack Type Test
# ============================================================================

def test_get_attack_type():
    """Test get_attack_type function"""
    print("\n🔧 Testing Get Attack Type...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import get_attack_type, ATTACK_TYPES
        
        # Test 1: Valid attack type
        prompt_injection = get_attack_type("prompt_injection")
        print(f"   Attack type: prompt_injection")
        print(f"   Attacks: {len(prompt_injection)}")
        
        assert prompt_injection == ATTACK_TYPES["prompt_injection"], \
            "Should return prompt_injection attacks"
        assert len(prompt_injection) > 0, "Should have at least one attack"
        
        # Test 2: Another valid attack type
        data_exfiltration = get_attack_type("data_exfiltration")
        print(f"   Attack type: data_exfiltration")
        print(f"   Attacks: {len(data_exfiltration)}")
        
        assert data_exfiltration == ATTACK_TYPES["data_exfiltration"], \
            "Should return data_exfiltration attacks"
        
        # Test 3: Invalid attack type (should return empty list)
        invalid = get_attack_type("invalid_type")
        print(f"   Attack type: invalid_type")
        print(f"   Attacks: {invalid}")
        
        assert invalid == [], "Should return empty list for invalid type"
        print(f"   ✅ Get attack type works correctly")
        print(f"   ⚠️  ISSUE: Returns empty list for unknown types (no error)")
        print(f"   ⚠️  Expected: Raise ValueError or return None")
        print(f"   ⚠️  Actual: Returns empty list (line 61)")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ List Attack Types Test
# ============================================================================

def test_list_attack_types():
    """Test list_attack_types function"""
    print("\n🔧 Testing List Attack Types...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import list_attack_types, ATTACK_TYPES
        
        attack_types_list = list_attack_types()
        
        print(f"   Attack types: {attack_types_list}")
        
        assert isinstance(attack_types_list, list), "Should return a list"
        assert len(attack_types_list) == len(ATTACK_TYPES), \
            "Should return all attack types"
        assert attack_types_list == list(ATTACK_TYPES.keys()), \
            "Should return keys from ATTACK_TYPES"
        
        # Check expected types
        expected_types = [
            "prompt_injection",
            "data_exfiltration",
            "social_engineering",
            "obfuscation",
            "jailbreak",
            "poisoning",
            "privilege_escalation"
        ]
        
        for expected_type in expected_types:
            assert expected_type in attack_types_list, \
                f"Should include {expected_type}"
        
        print(f"   ✅ List attack types works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Attack Content Quality Test
# ============================================================================

def test_attack_content_quality():
    """Test attack content quality and variety"""
    print("\n⚠️  Testing Attack Content Quality...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test each attack type has content
        for attack_type, attacks in ATTACK_TYPES.items():
            print(f"   {attack_type}:")
            print(f"     Count: {len(attacks)}")
            
            assert len(attacks) > 0, f"{attack_type} should have at least one attack"
            
            # Check attack lengths
            lengths = [len(attack) for attack in attacks]
            print(f"     Length range: {min(lengths)}-{max(lengths)} chars")
            print(f"     Average length: {sum(lengths) / len(lengths):.1f} chars")
            
            # Check for variety (at least 2 different attacks)
            if len(attacks) > 1:
                unique_attacks = len(set(attacks))
                print(f"     Unique attacks: {unique_attacks}")
                
                if unique_attacks < len(attacks):
                    print(f"     ⚠️  ISSUE: Duplicate attacks found")
                elif unique_attacks == 1:
                    print(f"     ⚠️  ISSUE: Only one unique attack (needs variety)")
            
            # Show example
            if len(attacks) > 0:
                print(f"     Example: {attacks[0][:80]}...")
        
        print(f"   ✅ Attack content quality checked")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Attack Type Coverage Test
# ============================================================================

def test_attack_type_coverage():
    """Test that all attack types have reasonable coverage"""
    print("\n🔧 Testing Attack Type Coverage...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Minimum attacks per type
        MIN_ATTACKS_PER_TYPE = 3
        
        print(f"   Checking coverage (min {MIN_ATTACKS_PER_TYPE} attacks per type)...")
        
        low_coverage = []
        for attack_type, attacks in ATTACK_TYPES.items():
            count = len(attacks)
            print(f"   {attack_type}: {count} attacks")
            
            if count < MIN_ATTACKS_PER_TYPE:
                low_coverage.append((attack_type, count))
                print(f"     ⚠️  ISSUE: Low coverage ({count} < {MIN_ATTACKS_PER_TYPE})")
        
        if low_coverage:
            print(f"   ⚠️  ISSUE: Some attack types have low coverage:")
            for attack_type, count in low_coverage:
                print(f"     - {attack_type}: {count} attacks")
        else:
            print(f"   ✅ All attack types have sufficient coverage")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Edge Cases Test
# ============================================================================

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import get_attack_type, list_attack_types
        
        # Test 1: None input
        try:
            result = get_attack_type(None)
            print(f"   None input handled: {result}")
            assert result == [], "Should return empty list for None"
        except Exception as e:
            print(f"   ⚠️  None input raised error: {type(e).__name__}: {e}")
        
        # Test 2: Empty string
        result = get_attack_type("")
        print(f"   Empty string input: {result}")
        assert result == [], "Should return empty list for empty string"
        
        # Test 3: Case sensitivity
        result_lower = get_attack_type("prompt_injection")
        result_upper = get_attack_type("PROMPT_INJECTION")
        print(f"   Case sensitivity: lower={len(result_lower)}, upper={len(result_upper)}")
        
        # Should be case sensitive (dictionary lookup)
        assert result_lower != result_upper or result_upper == [], \
            "Should be case sensitive or return empty for invalid case"
        
        print(f"   ✅ Edge cases handled")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Attack Catalog Immutability Test
# ============================================================================

def test_attack_catalog_immutability():
    """Test if attack catalog can be modified (should be protected)"""
    print("\n⚠️  Testing Attack Catalog Immutability...")
    reset_env()
    
    try:
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test if catalog can be modified directly
        original_count = len(ATTACK_TYPES)
        
        # Try to add a new attack type (should work since it's a dict)
        ATTACK_TYPES["test_type"] = ["test attack"]
        
        if len(ATTACK_TYPES) > original_count:
            print(f"   ⚠️  ISSUE: Catalog can be modified directly")
            print(f"   ⚠️  Expected: Immutable catalog or protected access")
            print(f"   ⚠️  Actual: Catalog is mutable dict")
            print(f"   ⚠️  Impact: Attacks can be changed at runtime")
            
            # Clean up
            del ATTACK_TYPES["test_type"]
        else:
            print(f"   ✅ Catalog is protected from modification")
        
        # Check if there's a way to protect catalog
        print(f"   ⚠️  ISSUE: No catalog protection mechanism")
        print(f"   ⚠️  Expected: Read-only access or validation before changes")
        print(f"   ⚠️  Actual: Catalog is regular dict (mutable)")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("VecSec Attack Catalog Functional Diagnostics")
    print("=" * 60)
    
    test_attack_types()
    test_hardcoded_attacks()
    test_get_attack_type()
    test_list_attack_types()
    test_attack_content_quality()
    test_attack_type_coverage()
    test_edge_cases()
    test_attack_catalog_immutability()
    
    print("\n" + "=" * 60)
    print("🏁 Attack Catalog Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Hardcoded attacks (not configurable)")
    print("   ⚠️  HIGH: Returns empty list for unknown types (no error)")
    print("   ⚠️  MEDIUM: Catalog is mutable (no protection)")
    print("   ⚠️  MEDIUM: No validation functions")


