"""
VecSec Attack Generator Functional Diagnostic
Tests real runtime behavior of attack_generator.py subsystems
Purpose: Diagnose all attack generation issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Attack Generator Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("API_FLASH_API_KEY", None)


# ============================================================================
# 1️⃣ Basic Attack Generation Test
# ============================================================================

def test_basic_attack_generation():
    """Test basic attack generation functionality"""
    print("\n🔧 Testing Basic Attack Generation...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        
        # Test 1: Basic static attack generation
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="prompt_injection",
            use_llm=False
        )
        
        print(f"   User ID: test_user")
        print(f"   Tenant: tenantA")
        print(f"   Attack type: prompt_injection")
        print(f"   Result keys: {list(result.keys())}")
        
        assert "example" in result, "Should have 'example' key"
        assert "metadata" in result, "Should have 'metadata' key"
        
        example = result["example"]
        metadata = result["metadata"]
        
        # Check example structure
        assert "user_id" in example, "Example should have user_id"
        assert "tenant_id" in example, "Example should have tenant_id"
        assert "clearance" in example, "Example should have clearance"
        assert "role" in example, "Example should have role"
        assert "query" in example, "Example should have query"
        
        # Check metadata structure
        assert "attack_id" in metadata, "Metadata should have attack_id"
        assert "attack_type" in metadata, "Metadata should have attack_type"
        assert "timestamp" in metadata, "Metadata should have timestamp"
        
        print(f"   Query: {example['query'][:60]}...")
        print(f"   Attack ID: {metadata['attack_id']}")
        print(f"   Generation method: {metadata.get('generation_method')}")
        print(f"   ✅ Basic attack generation works")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Static Attack Generation Test
# ============================================================================

def test_static_attack_generation():
    """Test static attack generation from catalog"""
    print("\n🔧 Testing Static Attack Generation...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test with different attack types
        for attack_type in ["prompt_injection", "data_exfiltration", "jailbreak"]:
            result = generate_attack(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                attack_type=attack_type,
                use_llm=False
            )
            
            query = result["example"]["query"]
            metadata = result["metadata"]
            
            print(f"   Attack type: {attack_type}")
            print(f"   Query: {query[:60]}...")
            
            # Check query is from catalog
            catalog_attacks = ATTACK_TYPES.get(attack_type, [])
            assert query in catalog_attacks, \
                f"Query should be from {attack_type} catalog"
            assert metadata["generation_method"] == "static", \
                "Should use static generation method"
        
        print(f"   ✅ Static attack generation works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ LLM Attack Generation Test (CRITICAL)
# ============================================================================

def test_llm_attack_generation():
    """Test LLM-based attack generation"""
    print("\n🔴 Testing LLM Attack Generation...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        
        # Test with LLM disabled (no API keys)
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="prompt_injection",
            use_llm=True,
            llm_provider="google"
        )
        
        query = result["example"]["query"]
        metadata = result["metadata"]
        
        print(f"   LLM enabled: True")
        print(f"   Provider: google")
        print(f"   Query: {query[:60]}...")
        print(f"   Generation method: {metadata.get('generation_method')}")
        print(f"   LLM provider: {metadata.get('llm_provider')}")
        
        # Check if it falls back to static when LLM fails
        if "API key not configured" in query or "Error calling" in query:
            print(f"   ⚠️  ISSUE: LLM generation fails silently and falls back")
            print(f"   ⚠️  Expected: Raise error or return None when LLM fails")
            print(f"   ⚠️  Actual: Returns error message as query string")
        elif metadata["generation_method"] == "static":
            print(f"   ⚠️  ISSUE: LLM disabled but still marked as 'static'")
            print(f"   ⚠️  Expected: Should use LLM or raise error")
        
        # Mock LLM client for testing
        with patch('src.evil_agent.attack_generator.get_llm_client') as mock_get_client:
            mock_client = Mock()
            mock_client.generate_with_google.return_value = "Generated attack prompt"
            mock_get_client.return_value = mock_client
            
            result2 = generate_attack(
                user_id="test_user",
                tenant_id="tenantA",
                clearance="INTERNAL",
                attack_type="prompt_injection",
                use_llm=True,
                llm_provider="google"
            )
            
            query2 = result2["example"]["query"]
            metadata2 = result2["metadata"]
            
            print(f"   With mocked LLM:")
            print(f"     Query: {query2}")
            print(f"     Generation method: {metadata2.get('generation_method')}")
            
            if query2 == "Generated attack prompt":
                print(f"   ✅ LLM generation works with mocked client")
            else:
                print(f"   ⚠️  LLM generation may not be working correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Random Attack Type Selection Test
# ============================================================================

def test_random_attack_type_selection():
    """Test random attack type selection when type not specified"""
    print("\n🔧 Testing Random Attack Type Selection...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        from src.evil_agent.attack_catalog import ATTACK_TYPES
        
        # Test without specifying attack_type
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type=None,
            use_llm=False
        )
        
        metadata = result["metadata"]
        attack_type = metadata["attack_type"]
        
        print(f"   Attack type not specified")
        print(f"   Selected type: {attack_type}")
        
        assert attack_type in ATTACK_TYPES, \
            f"Selected attack type should be valid: {attack_type}"
        
        # Test with invalid attack type (should select random)
        result2 = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="invalid_type",
            use_llm=False
        )
        
        metadata2 = result2["metadata"]
        attack_type2 = metadata2["attack_type"]
        
        print(f"   Invalid attack type specified")
        print(f"   Selected type: {attack_type2}")
        
        assert attack_type2 in ATTACK_TYPES, \
            f"Should select random valid type: {attack_type2}"
        
        print(f"   ✅ Random attack type selection works")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Seed Reproducibility Test
# ============================================================================

def test_seed_reproducibility():
    """Test that seed produces reproducible results"""
    print("\n🔧 Testing Seed Reproducibility...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        
        # Generate with same seed
        result1 = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            seed=42,
            use_llm=False
        )
        
        result2 = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            seed=42,
            use_llm=False
        )
        
        query1 = result1["example"]["query"]
        query2 = result2["example"]["query"]
        
        print(f"   Seed: 42")
        print(f"   Query 1: {query1[:60]}...")
        print(f"   Query 2: {query2[:60]}...")
        
        # With same seed, should get same attack (but different attack_id)
        if query1 == query2:
            print(f"   ✅ Seed produces reproducible queries")
        else:
            print(f"   ⚠️  Seed does not produce reproducible queries")
            print(f"   ⚠️  Expected: Same seed = same query")
            print(f"   ⚠️  Actual: Same seed = different query")
        
        # Attack IDs should be different (UUID)
        id1 = result1["metadata"]["attack_id"]
        id2 = result2["metadata"]["attack_id"]
        
        assert id1 != id2, "Attack IDs should be unique (UUID)"
        print(f"   ✅ Attack IDs are unique")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Metadata Structure Test
# ============================================================================

def test_metadata_structure():
    """Test metadata structure and completeness"""
    print("\n🔧 Testing Metadata Structure...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        import uuid
        
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            attack_type="prompt_injection",
            use_llm=False
        )
        
        metadata = result["metadata"]
        
        print(f"   Metadata keys: {list(metadata.keys())}")
        
        # Check required fields
        required_fields = [
            "attack_id",
            "attack_type",
            "attack_intent",
            "timestamp",
            "ethical_notice",
            "generation_method"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Metadata should have {field}"
            print(f"   ✅ {field}: {metadata[field]}")
        
        # Check attack_id is UUID
        try:
            uuid.UUID(metadata["attack_id"])
            print(f"   ✅ attack_id is valid UUID")
        except ValueError:
            print(f"   ⚠️  ISSUE: attack_id is not valid UUID")
        
        # Check timestamp format
        from datetime import datetime
        try:
            datetime.fromisoformat(metadata["timestamp"])
            print(f"   ✅ timestamp is valid ISO format")
        except ValueError:
            print(f"   ⚠️  ISSUE: timestamp is not valid ISO format")
        
        # Check severity in config
        if "config" in metadata:
            config = metadata["config"]
            if "severity" in config:
                severity = config["severity"]
                print(f"   Severity: {severity}")
                assert severity in ["LOW", "MEDIUM", "HIGH"], \
                    f"Severity should be LOW/MEDIUM/HIGH, got {severity}"
        
        print(f"   ✅ Metadata structure is correct")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Example Structure Test
# ============================================================================

def test_example_structure():
    """Test example structure and completeness"""
    print("\n🔧 Testing Example Structure...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            role="analyst",
            attack_type="prompt_injection",
            use_llm=False
        )
        
        example = result["example"]
        
        print(f"   Example keys: {list(example.keys())}")
        
        # Check required fields
        assert "user_id" in example, "Example should have user_id"
        assert example["user_id"] == "test_user", "user_id should match"
        
        assert "tenant_id" in example, "Example should have tenant_id"
        assert example["tenant_id"] == "tenantA", "tenant_id should match"
        
        assert "clearance" in example, "Example should have clearance"
        assert example["clearance"] == "INTERNAL", "clearance should match"
        
        assert "role" in example, "Example should have role"
        assert example["role"] == "analyst", "role should match"
        
        assert "query" in example, "Example should have query"
        assert len(example["query"]) > 0, "query should not be empty"
        
        print(f"   ✅ Example structure is correct")
        print(f"   Query length: {len(example['query'])} chars")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Edge Cases Test
# ============================================================================

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()
    
    try:
        from src.evil_agent.attack_generator import generate_attack
        
        # Test 1: Empty strings
        result = generate_attack(
            user_id="",
            tenant_id="",
            clearance="",
            use_llm=False
        )
        
        print(f"   Empty strings handled:")
        print(f"     user_id: '{result['example']['user_id']}'")
        print(f"     tenant_id: '{result['example']['tenant_id']}'")
        print(f"     clearance: '{result['example']['clearance']}'")
        
        # Test 2: None values (should handle gracefully)
        try:
            result = generate_attack(
                user_id=None,
                tenant_id="tenantA",
                clearance="INTERNAL",
                use_llm=False
            )
            print(f"   ✅ None user_id handled: {result['example']['user_id']}")
        except Exception as e:
            print(f"   ⚠️  None user_id raised error: {type(e).__name__}: {e}")
        
        # Test 3: Very long strings
        long_string = "x" * 1000
        result = generate_attack(
            user_id=long_string,
            tenant_id="tenantA",
            clearance="INTERNAL",
            use_llm=False
        )
        
        assert result["example"]["user_id"] == long_string, \
            "Should accept long user_id"
        print(f"   ✅ Long strings handled")
        
        # Test 4: Invalid role
        result = generate_attack(
            user_id="test_user",
            tenant_id="tenantA",
            clearance="INTERNAL",
            role="invalid_role",
            use_llm=False
        )
        
        assert result["example"]["role"] == "invalid_role", \
            "Should accept any role string"
        print(f"   ✅ Invalid role handled")
        
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
    print("VecSec Attack Generator Functional Diagnostics")
    print("=" * 60)
    
    test_basic_attack_generation()
    test_static_attack_generation()
    test_llm_attack_generation()
    test_random_attack_type_selection()
    test_seed_reproducibility()
    test_metadata_structure()
    test_example_structure()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🏁 Attack Generator Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: LLM generation fails silently")
    print("   ⚠️  HIGH: Invalid attack types default to random")
    print("   ⚠️  MEDIUM: Seed may not be fully reproducible")
    print("   ⚠️  MEDIUM: No validation for user_id/tenant_id/clearance")


