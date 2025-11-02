"""
VecSec Policy Manager Functional Diagnostic
Tests real runtime behavior of policy_manager.py subsystems
Purpose: Diagnose all policy management issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Policy Manager Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables"""
    os.environ.pop("USE_CHROMA", None)
    os.environ.pop("CHROMA_PATH", None)


# ============================================================================
# 1️⃣ Policy Constants Test
# ============================================================================

def test_policy_constants():
    """Test that policy constants are defined"""
    print("\n🔧 Testing Policy Constants...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import TENANT_POLICIES, ROLE_POLICIES
        
        # Test TENANT_POLICIES
        print(f"   TENANT_POLICIES keys: {list(TENANT_POLICIES.keys())}")
        assert "tenantA" in TENANT_POLICIES, "Should have tenantA"
        assert "tenantB" in TENANT_POLICIES, "Should have tenantB"
        
        # Check tenantA policy structure
        tenantA_policy = TENANT_POLICIES["tenantA"]
        print(f"   tenantA policy: {list(tenantA_policy.keys())}")
        assert "clearance" in tenantA_policy, "Should have clearance"
        assert "topics" in tenantA_policy, "Should have topics"
        assert "sensitivity" in tenantA_policy, "Should have sensitivity"
        
        # Test ROLE_POLICIES
        print(f"   ROLE_POLICIES keys: {list(ROLE_POLICIES.keys())}")
        assert "admin" in ROLE_POLICIES, "Should have admin"
        assert "superuser" in ROLE_POLICIES, "Should have superuser"
        assert "analyst" in ROLE_POLICIES, "Should have analyst"
        assert "guest" in ROLE_POLICIES, "Should have guest"
        
        # Check admin policy structure
        admin_policy = ROLE_POLICIES["admin"]
        print(f"   admin policy: {list(admin_policy.keys())}")
        assert "allowed_operations" in admin_policy, "Should have allowed_operations"
        assert "max_clearance" in admin_policy, "Should have max_clearance"
        assert "cross_tenant_access" in admin_policy, "Should have cross_tenant_access"
        assert "bypass_restrictions" in admin_policy, "Should have bypass_restrictions"
        
        print(f"   ✅ Policy constants defined correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Hardcoded Policies Test (CRITICAL)
# ============================================================================

def test_hardcoded_policies():
    """Test that policies are hardcoded (not configurable)"""
    print("\n🔴 Testing Hardcoded Policies...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import TENANT_POLICIES, ROLE_POLICIES
        
        # Test tenant policies
        print(f"   Tenant policies: {len(TENANT_POLICIES)} tenants")
        for tenant_id, policy in TENANT_POLICIES.items():
            print(f"   Tenant: {tenant_id}")
            print(f"     Clearance: {policy.get('clearance')}")
            print(f"     Sensitivity: {policy.get('sensitivity')}")
            print(f"     Topics: {policy.get('topics', [])}")
        
        # Test role policies
        print(f"   Role policies: {len(ROLE_POLICIES)} roles")
        for role, policy in ROLE_POLICIES.items():
            print(f"   Role: {role}")
            print(f"     Max clearance: {policy.get('max_clearance')}")
            print(f"     Allowed operations: {policy.get('allowed_operations', [])}")
            print(f"     Cross-tenant access: {policy.get('cross_tenant_access')}")
        
        print(f"   ⚠️  ISSUE: Policies are hardcoded (not configurable)")
        print(f"   ⚠️  Expected: Configurable via file/DB/API")
        print(f"   ⚠️  Actual: Hardcoded in Python code (lines 6-37)")
        print(f"   ⚠️  Impact: Cannot change policies without code changes")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Get Tenant Policy Test
# ============================================================================

def test_get_tenant_policy():
    """Test get_tenant_policy function"""
    print("\n🔧 Testing Get Tenant Policy...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import get_tenant_policy, TENANT_POLICIES
        
        # Test 1: Valid tenant
        policy1 = get_tenant_policy("tenantA")
        print(f"   Tenant: tenantA")
        print(f"   Policy: {policy1}")
        
        assert policy1 == TENANT_POLICIES["tenantA"], "Should return tenantA policy"
        assert "clearance" in policy1, "Should have clearance"
        
        # Test 2: Another valid tenant
        policy2 = get_tenant_policy("tenantB")
        print(f"   Tenant: tenantB")
        print(f"   Policy: {policy2}")
        
        assert policy2 == TENANT_POLICIES["tenantB"], "Should return tenantB policy"
        
        # Test 3: Invalid tenant (should return empty dict)
        policy3 = get_tenant_policy("invalid_tenant")
        print(f"   Tenant: invalid_tenant")
        print(f"   Policy: {policy3}")
        
        assert policy3 == {}, "Should return empty dict for invalid tenant"
        
        print(f"   ✅ Get tenant policy works correctly")
        print(f"   ⚠️  ISSUE: Returns empty dict for unknown tenants (no error)")
        print(f"   ⚠️  Expected: Raise ValueError or return None")
        print(f"   ⚠️  Actual: Returns empty dict (line 42)")
        print(f"   ⚠️  Impact: Silent failures, need to check for empty dict")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Get Role Policy Test
# ============================================================================

def test_get_role_policy():
    """Test get_role_policy function"""
    print("\n🔧 Testing Get Role Policy...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import get_role_policy, ROLE_POLICIES
        
        # Test 1: Valid roles
        for role in ["admin", "superuser", "analyst", "guest"]:
            policy = get_role_policy(role)
            print(f"   Role: {role}")
            print(f"     Max clearance: {policy.get('max_clearance')}")
            print(f"     Operations: {policy.get('allowed_operations', [])}")
            
            assert policy == ROLE_POLICIES[role], f"Should return {role} policy"
        
        # Test 2: Invalid role (should default to guest)
        policy_invalid = get_role_policy("invalid_role")
        print(f"   Role: invalid_role")
        print(f"   Policy: {policy_invalid}")
        print(f"   Default to guest: {policy_invalid == ROLE_POLICIES['guest']}")
        
        assert policy_invalid == ROLE_POLICIES["guest"], "Should default to guest"
        
        print(f"   ✅ Get role policy works correctly")
        print(f"   ⚠️  ISSUE: Unknown roles default to guest (silent fallback)")
        print(f"   ⚠️  Expected: Raise ValueError or return None")
        print(f"   ⚠️  Actual: Returns guest policy (line 47)")
        print(f"   ⚠️  Impact: Invalid roles silently treated as guest")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Clearance Level Hierarchy Test
# ============================================================================

def test_clearance_hierarchy():
    """Test clearance level hierarchy"""
    print("\n⚠️  Testing Clearance Level Hierarchy...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import ROLE_POLICIES
        
        # Define clearance hierarchy
        clearance_levels = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SECRET"]
        
        print(f"   Clearance hierarchy (low to high): {clearance_levels}")
        
        # Check role max clearances
        print(f"   Role max clearances:")
        for role, policy in ROLE_POLICIES.items():
            max_clearance = policy.get("max_clearance")
            clearance_level = clearance_levels.index(max_clearance) if max_clearance in clearance_levels else -1
            print(f"     {role}: {max_clearance} (level {clearance_level})")
            
            assert max_clearance in clearance_levels, f"{role} has invalid clearance: {max_clearance}"
        
        # Check if hierarchy is correct
        admin_clearance = ROLE_POLICIES["admin"]["max_clearance"]
        superuser_clearance = ROLE_POLICIES["superuser"]["max_clearance"]
        analyst_clearance = ROLE_POLICIES["analyst"]["max_clearance"]
        guest_clearance = ROLE_POLICIES["guest"]["max_clearance"]
        
        admin_level = clearance_levels.index(admin_clearance)
        superuser_level = clearance_levels.index(superuser_clearance)
        analyst_level = clearance_levels.index(analyst_clearance)
        guest_level = clearance_levels.index(guest_clearance)
        
        print(f"   Clearance levels:")
        print(f"     admin: {admin_clearance} (level {admin_level})")
        print(f"     superuser: {superuser_clearance} (level {superuser_level})")
        print(f"     analyst: {analyst_clearance} (level {analyst_level})")
        print(f"     guest: {guest_clearance} (level {guest_level})")
        
        # Check hierarchy (admin > superuser > analyst > guest)
        assert admin_level > superuser_level, "Admin should have higher clearance than superuser"
        assert superuser_level > analyst_level, "Superuser should have higher clearance than analyst"
        assert analyst_level > guest_level, "Analyst should have higher clearance than guest"
        
        print(f"   ✅ Clearance hierarchy is correct")
        
        print(f"   ⚠️  ISSUE: No explicit hierarchy comparison function")
        print(f"   ⚠️  Expected: can_access_clearance(role, required_clearance) function")
        print(f"   ⚠️  Actual: Manual comparison needed")
        print(f"   ⚠️  Impact: Need to implement clearance checks manually")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Role Permissions Test
# ============================================================================

def test_role_permissions():
    """Test role permissions and restrictions"""
    print("\n🔧 Testing Role Permissions...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import ROLE_POLICIES
        
        # Test each role's permissions
        for role, policy in ROLE_POLICIES.items():
            print(f"   Role: {role}")
            
            operations = policy.get("allowed_operations", [])
            print(f"     Operations: {operations}")
            
            cross_tenant = policy.get("cross_tenant_access", False)
            print(f"     Cross-tenant access: {cross_tenant}")
            
            bypass_restrictions = policy.get("bypass_restrictions", [])
            print(f"     Bypass restrictions: {bypass_restrictions}")
            
            # Admin should have all operations
            if role == "admin":
                assert "read" in operations
                assert "write" in operations
                assert "delete" in operations
                assert cross_tenant == True, "Admin should have cross-tenant access"
                assert "topic_scope" in bypass_restrictions, "Admin should bypass topic scope"
            
            # Guest should only have read
            if role == "guest":
                assert operations == ["read"], "Guest should only have read"
                assert cross_tenant == False, "Guest should not have cross-tenant access"
                assert bypass_restrictions == [], "Guest should not bypass any restrictions"
        
        print(f"   ✅ Role permissions are correct")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Tenant Topic Access Test
# ============================================================================

def test_tenant_topic_access():
    """Test tenant topic access policies"""
    print("\n🔧 Testing Tenant Topic Access...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import TENANT_POLICIES
        
        # Test each tenant's allowed topics
        for tenant_id, policy in TENANT_POLICIES.items():
            print(f"   Tenant: {tenant_id}")
            
            topics = policy.get("topics", [])
            print(f"     Allowed topics: {topics}")
            
            clearance = policy.get("clearance")
            print(f"     Clearance: {clearance}")
            
            sensitivity = policy.get("sensitivity")
            print(f"     Sensitivity: {sensitivity}")
            
            # Check if topics are defined
            assert len(topics) > 0, f"{tenant_id} should have at least one topic"
            
            # tenantA should have retrieval/RAG topics
            if tenant_id == "tenantA":
                assert "retrieval" in topics or "RAG" in topics, "tenantA should have RAG topics"
            
            # tenantB should have finance topics
            if tenant_id == "tenantB":
                assert "finance" in topics, "tenantB should have finance topics"
        
        print(f"   ✅ Tenant topic access is correct")
        
        print(f"   ⚠️  ISSUE: No topic validation function")
        print(f"   ⚠️  Expected: can_access_topic(tenant, topic) function")
        print(f"   ⚠️  Actual: Manual list checking needed")
        print(f"   ⚠️  Impact: Need to implement topic access checks manually")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Policy Validation Test
# ============================================================================

def test_policy_validation():
    """Test policy validation (should be missing)"""
    print("\n⚠️  Testing Policy Validation...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import TENANT_POLICIES, ROLE_POLICIES
        
        # Check if validation functions exist
        validation_functions = [
            "validate_tenant_policy",
            "validate_role_policy",
            "can_access_clearance",
            "can_access_topic",
            "can_access_tenant"
        ]
        
        from src.sec_agent import policy_manager
        
        missing_functions = []
        for func_name in validation_functions:
            if not hasattr(policy_manager, func_name):
                missing_functions.append(func_name)
        
        print(f"   Missing validation functions: {missing_functions}")
        
        if missing_functions:
            print(f"   ⚠️  ISSUE: No policy validation functions")
            print(f"   ⚠️  Expected: Validation functions for policy checks")
            print(f"   ⚠️  Actual: No validation functions (manual checks only)")
            print(f"   ⚠️  Impact: Need to implement validation logic manually")
        else:
            print(f"   ✅ All validation functions present")
        
        # Test policy structure validation
        print(f"   Testing policy structure...")
        
        # Check tenant policies have required keys
        for tenant_id, policy in TENANT_POLICIES.items():
            required_keys = ["clearance", "topics", "sensitivity"]
            missing_keys = [key for key in required_keys if key not in policy]
            
            if missing_keys:
                print(f"   ⚠️  Tenant {tenant_id} missing keys: {missing_keys}")
            else:
                print(f"   ✅ Tenant {tenant_id} has all required keys")
        
        # Check role policies have required keys
        for role, policy in ROLE_POLICIES.items():
            required_keys = ["allowed_operations", "max_clearance", "cross_tenant_access", "bypass_restrictions"]
            missing_keys = [key for key in required_keys if key not in policy]
            
            if missing_keys:
                print(f"   ⚠️  Role {role} missing keys: {missing_keys}")
            else:
                print(f"   ✅ Role {role} has all required keys")
        
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
        from src.sec_agent.policy_manager import get_tenant_policy, get_role_policy
        
        # Test 1: None input
        try:
            policy = get_tenant_policy(None)
            print(f"   None tenant handled: {policy}")
            assert policy == {}, "Should return empty dict for None"
        except Exception as e:
            print(f"   ⚠️  None tenant raised error: {type(e).__name__}: {e}")
        
        # Test 2: Empty string
        policy = get_tenant_policy("")
        print(f"   Empty string tenant: {policy}")
        assert policy == {}, "Should return empty dict for empty string"
        
        # Test 3: None role
        try:
            policy = get_role_policy(None)
            print(f"   None role handled: {policy}")
            # Should default to guest or raise error
        except Exception as e:
            print(f"   None role raised error: {type(e).__name__}: {e}")
        
        # Test 4: Empty string role
        policy = get_role_policy("")
        print(f"   Empty string role: {policy}")
        # Should default to guest or raise error
        
        print(f"   ✅ Edge cases handled")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 🔟 Policy Immutability Test
# ============================================================================

def test_policy_immutability():
    """Test if policies can be modified (should be protected)"""
    print("\n⚠️  Testing Policy Immutability...")
    reset_env()
    
    try:
        from src.sec_agent.policy_manager import TENANT_POLICIES, ROLE_POLICIES
        
        # Test if policies can be modified directly
        original_tenant_count = len(TENANT_POLICIES)
        original_role_count = len(ROLE_POLICIES)
        
        # Try to add a new tenant (should work since it's a dict)
        TENANT_POLICIES["test_tenant"] = {"clearance": "PUBLIC", "topics": [], "sensitivity": "PUBLIC"}
        
        if len(TENANT_POLICIES) > original_tenant_count:
            print(f"   ⚠️  ISSUE: Policies can be modified directly")
            print(f"   ⚠️  Expected: Immutable policies or protected access")
            print(f"   ⚠️  Actual: Policies are mutable dicts")
            print(f"   ⚠️  Impact: Policies can be changed at runtime (security risk)")
            
            # Clean up
            del TENANT_POLICIES["test_tenant"]
        else:
            print(f"   ✅ Policies are protected from modification")
        
        # Check if there's a way to protect policies
        print(f"   ⚠️  ISSUE: No policy protection mechanism")
        print(f"   ⚠️  Expected: Read-only access or validation before changes")
        print(f"   ⚠️  Actual: Policies are regular dicts (mutable)")
        
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
    print("VecSec Policy Manager Functional Diagnostics")
    print("=" * 60)
    
    test_policy_constants()
    test_hardcoded_policies()
    test_get_tenant_policy()
    test_get_role_policy()
    test_clearance_hierarchy()
    test_role_permissions()
    test_tenant_topic_access()
    test_policy_validation()
    test_edge_cases()
    test_policy_immutability()
    
    print("\n" + "=" * 60)
    print("🏁 Policy Manager Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Hardcoded policies (not configurable)")
    print("   ⚠️  HIGH: Returns empty dict for unknown tenants (no error)")
    print("   ⚠️  HIGH: Unknown roles default to guest (silent fallback)")
    print("   ⚠️  MEDIUM: No validation functions")
    print("   ⚠️  MEDIUM: Policies are mutable (no protection)")
    print("   ⚠️  MEDIUM: No explicit hierarchy comparison function")

