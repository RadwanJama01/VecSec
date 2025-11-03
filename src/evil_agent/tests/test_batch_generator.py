"""
VecSec Batch Generator Functional Diagnostic
Tests real runtime behavior of batch_generator.py subsystems
Purpose: Diagnose all batch generation issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Batch Generator Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)


# ============================================================================
# 1️⃣ Basic Batch Generation Test
# ============================================================================

def test_basic_batch_generation():
    """Test basic batch generation functionality"""
    print("\n🔧 Testing Basic Batch Generation...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test 1: Basic batch generation
        results = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            count_per_type=1,
            use_llm=False
        )
        
        print(f"   User ID: test_user")
        print(f"   Tenant: tenantA")
        print(f"   Count per type: 1")
        print(f"   Results: {len(results)} attacks")
        
        assert isinstance(results, list), "Should return a list"
        assert len(results) > 0, "Should generate at least one attack"
        
        # Check structure
        for result in results:
            assert "example" in result, "Each result should have 'example'"
            assert "metadata" in result, "Each result should have 'metadata'"
            assert result["metadata"]["attack_type"] is not None, \
                "Each result should have attack_type"
        
        print(f"   ✅ Basic batch generation works")
        print(f"   Generated {len(results)} attacks")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Batch Size Test
# ============================================================================

def test_batch_size():
    """Test batch generation with different sizes"""
    print("\n🔧 Testing Batch Size...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test with different counts
        for count in [1, 2, 3]:
            results = generate_batch(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                count_per_type=count,
                use_llm=False
            )
            
            print(f"   Count per type: {count}")
            print(f"   Total attacks: {len(results)}")
            
            # Should generate count * number_of_attack_types
            expected_count = count * len(ATTACK_TYPES)
            assert len(results) == expected_count, \
                f"Should generate {expected_count} attacks, got {len(results)}"
            
            print(f"   ✅ Generated {len(results)} attacks (expected {expected_count})")
        
        print(f"   ✅ Batch size works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Attack Type Selection Test
# ============================================================================

def test_attack_type_selection():
    """Test batch generation with specific attack types"""
    print("\n🔧 Testing Attack Type Selection...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test 1: All attack types (default)
        results1 = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_types=None,
            count_per_type=1,
            use_llm=False
        )
        
        print(f"   All attack types: {len(results1)} attacks")
        
        # Test 2: Specific attack types
        selected_types = ["prompt_injection", "data_exfiltration"]
        results2 = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_types=selected_types,
            count_per_type=1,
            use_llm=False
        )
        
        print(f"   Selected types {selected_types}: {len(results2)} attacks")
        
        assert len(results2) == len(selected_types), \
            f"Should generate {len(selected_types)} attacks, got {len(results2)}"
        
        # Check attack types
        generated_types = [r["metadata"]["attack_type"] for r in results2]
        print(f"   Generated types: {generated_types}")
        
        for attack_type in generated_types:
            assert attack_type in selected_types, \
                f"Attack type {attack_type} should be in selected types"
        
        print(f"   ✅ Attack type selection works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ LLM Batch Generation Test
# ============================================================================

def test_llm_batch_generation():
    """Test batch generation with LLM"""
    print("\n🔧 Testing LLM Batch Generation...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test with LLM disabled (no API keys)
        results = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            count_per_type=1,
            use_llm=True,
            llm_provider="google"
        )
        
        print(f"   LLM enabled: True")
        print(f"   Results: {len(results)} attacks")
        
        # Check generation methods
        methods = [r["metadata"]["generation_method"] for r in results]
        print(f"   Generation methods: {set(methods)}")
        
        # May fall back to static if LLM fails
        if "llm" in methods:
            print(f"   ✅ LLM generation used")
        elif "static" in methods:
            print(f"   ⚠️  LLM generation fell back to static")
            print(f"   ⚠️  Expected: Use LLM or raise error")
            print(f"   ⚠️  Actual: Falls back to static silently")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Role Parameter Test
# ============================================================================

def test_role_parameter():
    """Test role parameter in batch generation"""
    print("\n🔧 Testing Role Parameter...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test with different roles
        roles = ["admin", "analyst", "guest"]
        
        for role in roles:
            results = generate_batch(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                role=role,
                count_per_type=1,
                use_llm=False
            )
            
            # Check role in examples
            for result in results:
                example_role = result["example"]["role"]
                assert example_role == role, \
                    f"Role should be {role}, got {example_role}"
            
            print(f"   Role {role}: ✅ All attacks have correct role")
        
        print(f"   ✅ Role parameter works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Metadata Consistency Test
# ============================================================================

def test_metadata_consistency():
    """Test metadata consistency across batch"""
    print("\n🔧 Testing Metadata Consistency...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        results = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            count_per_type=2,
            use_llm=False
        )
        
        print(f"   Batch size: {len(results)}")
        
        # Check user_id consistency
        user_ids = [r["example"]["user_id"] for r in results]
        assert len(set(user_ids)) == 1, \
            "All attacks should have same user_id"
        assert user_ids[0] == "test_user", "user_id should match"
        
        # Check tenant_id consistency
        tenant_ids = [r["example"]["tenant_id"] for r in results]
        assert len(set(tenant_ids)) == 1, \
            "All attacks should have same tenant_id"
        assert tenant_ids[0] == "tenantA", "tenant_id should match"
        
        # Check clearance consistency
        clearances = [r["example"]["clearance"] for r in results]
        assert len(set(clearances)) == 1, \
            "All attacks should have same clearance"
        assert clearances[0] == "INTERNAL", "clearance should match"
        
        print(f"   ✅ Metadata consistency verified")
        print(f"     User IDs: {set(user_ids)}")
        print(f"     Tenant IDs: {set(tenant_ids)}")
        print(f"     Clearances: {set(clearances)}")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Empty Batch Test
# ============================================================================

def test_empty_batch():
    """Test batch generation with empty attack types"""
    print("\n🔧 Testing Empty Batch...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test with empty attack types list
        results = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_types=[],
            count_per_type=1,
            use_llm=False
        )
        
        print(f"   Empty attack types: {len(results)} attacks")
        
        # Should handle gracefully
        assert isinstance(results, list), "Should return a list"
        assert len(results) == 0, \
            "Should return empty list for empty attack types"
        
        print(f"   ✅ Empty batch handled correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Invalid Attack Types Test
# ============================================================================

def test_invalid_attack_types():
    """Test batch generation with invalid attack types"""
    print("\n⚠️  Testing Invalid Attack Types...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test with invalid attack types
        invalid_types = ["invalid_type_1", "invalid_type_2"]
        
        try:
            results = generate_batch(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                attack_types=invalid_types,
                count_per_type=1,
                use_llm=False
            )
            
            print(f"   Invalid attack types: {len(results)} attacks")
            
            # May generate attacks with random types or fail
            if len(results) == 0:
                print(f"   ✅ Returns empty list for invalid types")
            else:
                print(f"   ⚠️  ISSUE: Generates attacks even with invalid types")
                print(f"   ⚠️  Expected: Raise error or return empty list")
                print(f"   ⚠️  Actual: May generate attacks with default/random types")
        
        except Exception as e:
            print(f"   ✅ Raises error for invalid types: {type(e).__name__}")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 9️⃣ Edge Cases Test
# ============================================================================

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()
    
    try:
        from src.evil_agent.batch_generator import generate_batch
        
        # Test 1: Zero count
        results = generate_batch(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            count_per_type=0,
            use_llm=False
        )
        
        print(f"   Zero count: {len(results)} attacks")
        assert len(results) == 0, "Should return empty list for zero count"
        
        # Test 2: Negative count
        try:
            results = generate_batch(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                count_per_type=-1,
                use_llm=False
            )
            print(f"   Negative count: {len(results)} attacks")
            print(f"   ⚠️  ISSUE: Should not allow negative count")
        except Exception as e:
            print(f"   ✅ Raises error for negative count: {type(e).__name__}")
        
        # Test 3: None values
        try:
            results = generate_batch(
                user_id=None,
                tenant_id=None,
                clearance=None,
                count_per_type=1,
                use_llm=False
            )
            print(f"   None values: {len(results)} attacks")
            print(f"   ⚠️  Should handle None values or raise error")
        except Exception as e:
            print(f"   ✅ Raises error for None values: {type(e).__name__}")
        
        print(f"   ✅ Edge cases handled")
        
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
    print("VecSec Batch Generator Functional Diagnostics")
    print("=" * 60)
    
    test_basic_batch_generation()
    test_batch_size()
    test_attack_type_selection()
    test_llm_batch_generation()
    test_role_parameter()
    test_metadata_consistency()
    test_empty_batch()
    test_invalid_attack_types()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🏁 Batch Generator Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   ⚠️  HIGH: Invalid attack types may not be validated")
    print("   ⚠️  MEDIUM: LLM generation may fall back silently")
    print("   ⚠️  MEDIUM: No validation for user_id/tenant_id/clearance")
    print("   ⚠️  MEDIUM: Negative count may not be handled")


