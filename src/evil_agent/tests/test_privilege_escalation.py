"""
VecSec Privilege Escalation Functional Diagnostic
Tests real runtime behavior of privilege_escalation.py subsystems
Purpose: Diagnose all privilege escalation issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Privilege Escalation Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================


def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)


# ============================================================================
# 1️⃣ Basic Privilege Escalation Test
# ============================================================================


def test_basic_privilege_escalation():
    """Test basic privilege escalation generation"""
    print("\n🔧 Testing Basic Privilege Escalation...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Test 1: Basic privilege escalation
        result = generate_privilege_escalation_attack(
            user_id="attacker", tenant_id="tenantA", use_llm=False
        )

        print("   User ID: attacker")
        print("   Tenant: tenantA")
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
        assert "escalation_scenario" in metadata, "Metadata should have escalation_scenario"

        print(f"   Query: {example['query'][:60]}...")
        print(f"   Role: {example['role']}")
        print(f"   Clearance: {example['clearance']}")
        print("   ✅ Basic privilege escalation works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Hardcoded Escalation Scenarios Test (CRITICAL)
# ============================================================================


def test_hardcoded_escalation_scenarios():
    """Test that escalation scenarios are hardcoded"""
    print("\n🔴 Testing Hardcoded Escalation Scenarios...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Generate multiple attacks to see scenarios
        scenarios_seen = []
        for _ in range(20):
            result = generate_privilege_escalation_attack(
                user_id="attacker", tenant_id="tenantA", use_llm=False
            )

            scenario = result["metadata"]["escalation_scenario"]
            scenarios_seen.append(scenario["description"])

        unique_scenarios = set(scenarios_seen)
        print(f"   Generated attacks: {len(scenarios_seen)}")
        print(f"   Unique scenarios: {len(unique_scenarios)}")
        print(f"   Scenarios seen: {unique_scenarios}")

        # Check if scenarios are from hardcoded list
        print(
            f"   ⚠️  ISSUE: Escalation scenarios are hardcoded (only {len(unique_scenarios)} scenarios)"
        )
        print("   ⚠️  Expected: Algorithmic generation of ALL role/clearance/tenant combinations")
        print("   ⚠️  Actual: Hardcoded list with limited scenarios (line 39-46)")
        print("   ⚠️  Impact: Cannot test all possible privilege escalation paths")

        # Check expected scenarios from code
        expected_scenarios = [
            "Guest trying to access SECRET data",
            "Analyst trying to access CONFIDENTIAL data",
            "Guest trying to access CONFIDENTIAL data",
            "Analyst trying to access SECRET data",
            "Guest trying cross-tenant access",
            "Analyst trying cross-tenant access",
        ]

        print(f"   Expected scenarios: {len(expected_scenarios)}")
        for scenario in expected_scenarios:
            if scenario in unique_scenarios:
                print(f"     ✅ {scenario}")
            else:
                print(f"     ❌ Missing: {scenario}")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Scenario Types Test
# ============================================================================


def test_scenario_types():
    """Test different types of escalation scenarios"""
    print("\n🔧 Testing Scenario Types...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        target_clearance_scenarios = []
        target_tenant_scenarios = []

        for _ in range(30):
            result = generate_privilege_escalation_attack(
                user_id="attacker", tenant_id="tenantA", use_llm=False
            )

            scenario = result["metadata"]["escalation_scenario"]

            if "target_clearance" in scenario:
                target_clearance_scenarios.append(scenario)
            elif "target_tenant" in scenario:
                target_tenant_scenarios.append(scenario)

        print(f"   Target clearance scenarios: {len(target_clearance_scenarios)}")
        print(f"   Target tenant scenarios: {len(target_tenant_scenarios)}")

        # Check clearance escalations
        if target_clearance_scenarios:
            clearance_types = {}
            for s in target_clearance_scenarios:
                role = s.get("role")
                clearance = s.get("clearance")
                target = s.get("target_clearance")
                key = f"{role}:{clearance}->{target}"
                clearance_types[key] = clearance_types.get(key, 0) + 1

            print("   Clearance escalation patterns:")
            for pattern, count in clearance_types.items():
                print(f"     {pattern}: {count}")

        # Check tenant escalations
        if target_tenant_scenarios:
            tenant_types = {}
            for s in target_tenant_scenarios:
                role = s.get("role")
                target = s.get("target_tenant")
                key = f"{role}->{target}"
                tenant_types[key] = tenant_types.get(key, 0) + 1

            print("   Tenant escalation patterns:")
            for pattern, count in tenant_types.items():
                print(f"     {pattern}: {count}")

        print("   ✅ Scenario types identified")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Query Generation Test
# ============================================================================


def test_query_generation():
    """Test query generation for different scenarios"""
    print("\n🔧 Testing Query Generation...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        queries_seen = []

        for _ in range(20):
            result = generate_privilege_escalation_attack(
                user_id="attacker", tenant_id="tenantA", use_llm=False
            )

            query = result["example"]["query"]

            queries_seen.append(query)

            # Check query matches scenario
            role = result["example"]["role"]
            clearance = result["example"]["clearance"]

            assert role in query.lower() or clearance.lower() in query.lower(), (
                f"Query should mention role or clearance: {query}"
            )

        unique_queries = set(queries_seen)
        print(f"   Generated queries: {len(queries_seen)}")
        print(f"   Unique queries: {len(unique_queries)}")
        print(f"   Query variety: {len(unique_queries) / len(queries_seen) * 100:.1f}%")

        # Show example queries
        print("   Example queries:")
        for i, q in enumerate(list(unique_queries)[:3], 1):
            print(f"     {i}. {q[:60]}...")

        if len(unique_queries) < len(queries_seen) * 0.5:
            print("   ⚠️  ISSUE: Low query variety (repeated queries)")
            print("   ⚠️  Expected: More variety in generated queries")

        print("   ✅ Query generation works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Attack Type Parameter Test
# ============================================================================


def test_attack_type_parameter():
    """Test attack_type parameter"""
    print("\n🔧 Testing Attack Type Parameter...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Test with different attack types
        attack_types = [None, "data_exfiltration", "social_engineering"]

        for attack_type in attack_types:
            result = generate_privilege_escalation_attack(
                user_id="attacker", tenant_id="tenantA", attack_type=attack_type, use_llm=False
            )

            metadata = result["metadata"]
            result_attack_type = metadata["attack_type"]

            print(f"   Input attack_type: {attack_type}")
            print(f"   Result attack_type: {result_attack_type}")

            if attack_type is None:
                assert result_attack_type == "privilege_escalation", (
                    "Should default to 'privilege_escalation'"
                )
            else:
                assert result_attack_type == attack_type, (
                    f"Should use specified attack_type: {attack_type}"
                )

            print("   ✅ Attack type parameter works")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Seed Reproducibility Test
# ============================================================================


def test_seed_reproducibility():
    """Test that seed produces reproducible results"""
    print("\n🔧 Testing Seed Reproducibility...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Generate with same seed
        result1 = generate_privilege_escalation_attack(
            user_id="attacker", tenant_id="tenantA", seed=42, use_llm=False
        )

        result2 = generate_privilege_escalation_attack(
            user_id="attacker", tenant_id="tenantA", seed=42, use_llm=False
        )

        scenario1 = result1["metadata"]["escalation_scenario"]
        scenario2 = result2["metadata"]["escalation_scenario"]

        print("   Seed: 42")
        print(f"   Scenario 1: {scenario1['description']}")
        print(f"   Scenario 2: {scenario2['description']}")

        # With same seed, should get same scenario (but different attack_id)
        if scenario1 == scenario2:
            print("   ✅ Seed produces reproducible scenarios")
        else:
            print("   ⚠️  Seed does not produce reproducible scenarios")
            print("   ⚠️  Expected: Same seed = same scenario")
            print("   ⚠️  Actual: Same seed = different scenario")

        # Attack IDs should be different (UUID)
        id1 = result1["metadata"]["attack_id"]
        id2 = result2["metadata"]["attack_id"]

        assert id1 != id2, "Attack IDs should be unique (UUID)"
        print("   ✅ Attack IDs are unique")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Missing Combinations Test (CRITICAL)
# ============================================================================


def test_missing_combinations():
    """Test that not all combinations are generated"""
    print("\n🔴 Testing Missing Combinations...")
    reset_env()

    try:
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Generate many attacks to see all scenarios
        all_scenarios = []
        for _ in range(100):
            result = generate_privilege_escalation_attack(
                user_id="attacker", tenant_id="tenantA", use_llm=False
            )

            scenario = result["metadata"]["escalation_scenario"]
            scenario_key = (
                scenario.get("role"),
                scenario.get("clearance"),
                scenario.get("target_clearance"),
                scenario.get("target_tenant"),
            )
            all_scenarios.append(scenario_key)

        unique_scenarios = set(all_scenarios)
        print(f"   Generated attacks: {len(all_scenarios)}")
        print(f"   Unique scenarios: {len(unique_scenarios)}")

        # Calculate expected combinations
        # Role escalations: role -> higher role
        # Clearance escalations: clearance -> higher clearance
        # Tenant escalations: tenant -> other tenant

        print(f"   ⚠️  ISSUE: Only {len(unique_scenarios)} scenarios generated")
        print("   ⚠️  Expected: All possible role/clearance/tenant combinations")
        print("   ⚠️  Actual: Hardcoded list with 6 scenarios")
        print("   ⚠️  Missing combinations:")
        print("     - Superuser escalation attempts")
        print("     - Admin escalation attempts")
        print("     - All clearance level combinations")
        print("     - All tenant combinations")

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
        from src.evil_agent.privilege_escalation import generate_privilege_escalation_attack

        # Test 1: None values
        try:
            result = generate_privilege_escalation_attack(
                user_id=None, tenant_id=None, use_llm=False
            )
            print("   None values handled:")
            print(f"     user_id: {result['example']['user_id']}")
            print(f"     tenant_id: {result['example']['tenant_id']}")
        except Exception as e:
            print(f"   ⚠️  None values raised error: {type(e).__name__}: {e}")

        # Test 2: Empty strings
        result = generate_privilege_escalation_attack(user_id="", tenant_id="", use_llm=False)

        print("   Empty strings handled:")
        print(f"     user_id: '{result['example']['user_id']}'")
        print(f"     tenant_id: '{result['example']['tenant_id']}'")

        # Test 3: Invalid attack type
        result = generate_privilege_escalation_attack(
            user_id="attacker", tenant_id="tenantA", attack_type="invalid_type", use_llm=False
        )

        print(f"   Invalid attack type: {result['metadata']['attack_type']}")

        print("   ✅ Edge cases handled")

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
    print("VecSec Privilege Escalation Functional Diagnostics")
    print("=" * 60)

    test_basic_privilege_escalation()
    test_hardcoded_escalation_scenarios()
    test_scenario_types()
    test_query_generation()
    test_attack_type_parameter()
    test_seed_reproducibility()
    test_missing_combinations()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("🏁 Privilege Escalation Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Hardcoded escalation scenarios (only 6)")
    print("   🔴 CRITICAL: Missing many role/clearance/tenant combinations")
    print("   ⚠️  HIGH: Should be algorithmic to generate ALL combinations")
    print("   ⚠️  MEDIUM: Seed may not be fully reproducible")
    print("   ⚠️  MEDIUM: No validation for user_id/tenant_id")
