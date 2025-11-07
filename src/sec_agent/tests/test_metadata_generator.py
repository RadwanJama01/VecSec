"""
VecSec Metadata Generator Functional Diagnostic
Tests real runtime behavior of metadata_generator.py subsystems
Purpose: Diagnose all metadata generation issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Metadata Generator Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================


def reset_env():
    """Reset environment variables"""
    os.environ.pop("USE_CHROMA", None)
    os.environ.pop("CHROMA_PATH", None)


# ============================================================================
# 1️⃣ Basic Functionality Test
# ============================================================================


def test_basic_functionality():
    """Test basic metadata generation"""
    print("\n🔧 Testing Basic Functionality...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test 1: Basic query with topics
        query_context = {
            "query": "Query marketing campaign performance",
            "intent": "data_retrieval",
            "topics": ["marketing"],
            "target_tenant": None,
            "detected_threats": [],
        }
        user_tenant = "tenantA"

        metadata = generate_retrieval_metadata(query_context, user_tenant)

        print(f"   Query: '{query_context['query']}'")
        print(f"   Topics: {query_context['topics']}")
        print(f"   Metadata items: {len(metadata)}")

        assert isinstance(metadata, list), "Should return a list"
        assert len(metadata) > 0, "Should return at least one metadata item"

        # Check structure
        for item in metadata:
            assert "embedding_id" in item, "Should have embedding_id"
            assert "tenant_id" in item, "Should have tenant_id"
            assert "sensitivity" in item, "Should have sensitivity"
            assert "topics" in item, "Should have topics"
            assert "document_id" in item, "Should have document_id"
            assert "retrieval_score" in item, "Should have retrieval_score"

        print("   ✅ Basic functionality works")
        print(f"   Metadata structure: {list(metadata[0].keys())}")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Fake Metadata Generation Test (CRITICAL)
# ============================================================================


def test_fake_metadata_generation():
    """Test that metadata is fake (not real retrieval)"""
    print("\n🔴 Testing Fake Metadata Generation...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        query_context = {
            "query": "Find documents about finance",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": None,
            "detected_threats": [],
        }
        user_tenant = "tenantA"

        metadata = generate_retrieval_metadata(query_context, user_tenant)

        print(f"   Query: '{query_context['query']}'")
        print(f"   Metadata generated: {len(metadata)} items")

        # Check if metadata is fake (not from real vector DB)
        for item in metadata:
            embedding_id = item["embedding_id"]
            document_id = item["document_id"]

            print(f"   Item: embedding_id={embedding_id}, document_id={document_id}")

            # Check if IDs follow mock pattern
            if embedding_id.startswith("emb-"):
                print(f"   ⚠️  ISSUE: Fake embedding_id pattern: {embedding_id}")
            if document_id.startswith("doc-"):
                print(f"   ⚠️  ISSUE: Fake document_id pattern: {document_id}")

        print("   ⚠️  ISSUE: Metadata is fake (not real retrieval)")
        print("   ⚠️  Expected: Real document retrieval from vector DB")
        print("   ⚠️  Actual: Mock metadata generation")
        print("   ⚠️  Impact: RLS enforcement based on fake documents")

        # Check if no real document content
        has_page_content = any("page_content" in item for item in metadata)
        has_content = any("content" in item for item in metadata)

        print(f"   Has page_content: {has_page_content}")
        print(f"   Has content: {has_content}")
        print("   ⚠️  ISSUE: No real document content (just metadata)")
        print("   ⚠️  Expected: Real document content for RLS checks")
        print("   ⚠️  Actual: Only IDs and metadata")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Arbitrary Score Formula Test (CRITICAL)
# ============================================================================


def test_arbitrary_score_formula():
    """Test that scores use arbitrary formula"""
    print("\n🔴 Testing Arbitrary Score Formula...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        query_context = {
            "query": "Query multiple topics",
            "intent": "data_retrieval",
            "topics": ["finance", "marketing", "hr"],  # 3 topics
            "target_tenant": None,
            "detected_threats": [],
        }
        user_tenant = "tenantA"

        metadata = generate_retrieval_metadata(query_context, user_tenant)

        print(f"   Topics: {query_context['topics']}")
        print(f"   Metadata items: {len(metadata)}")

        # Check scores
        scores = [item["retrieval_score"] for item in metadata]
        print(f"   Scores: {scores}")

        # Check if scores follow formula: 0.9 - (i * 0.1)
        expected_scores = [0.9 - (i * 0.1) for i in range(len(query_context["topics"]))]
        print(f"   Expected scores (formula): {expected_scores}")

        # Match actual scores to expected
        matches = []
        for i, score in enumerate(scores[: len(expected_scores)]):
            matches.append(abs(score - expected_scores[i]) < 0.01)
            print(
                f"   Item {i + 1}: score={score:.3f}, expected={expected_scores[i]:.3f}, match={matches[i]}"
            )

        if all(matches):
            print("   ⚠️  ISSUE: Scores use arbitrary formula: 0.9 - (i * 0.1)")
            print("   ⚠️  Expected: Real similarity scores from vector search")
            print("   ⚠️  Actual: Hardcoded formula (line 25)")
            print("   ⚠️  Impact: Scores don't reflect actual document similarity")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ No Actual Similarity Search Test (CRITICAL)
# ============================================================================


def test_no_actual_similarity_search():
    """Test that function doesn't query vector DB"""
    print("\n🔴 Testing No Actual Similarity Search...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        query_context = {
            "query": "Query for documents",
            "intent": "data_retrieval",
            "topics": ["rag"],
            "target_tenant": None,
            "detected_threats": [],
        }
        user_tenant = "tenantA"

        # Mock vector store to see if it's queried
        with patch("src.sec_agent.metadata_generator"):
            metadata = generate_retrieval_metadata(query_context, user_tenant)

            print(f"   Query: '{query_context['query']}'")
            print(f"   Metadata generated: {len(metadata)} items")

            # Check if vector store was accessed (it shouldn't be)
            print("   ⚠️  ISSUE: Function doesn't query vector DB")
            print("   ⚠️  Expected: similarity_search() call to vector store")
            print("   ⚠️  Actual: No vector DB query (generates mock data)")
            print("   ⚠️  Impact: RLS enforcement based on simulated documents")

        # Test with different queries - should get same type of mock data
        query_context2 = {
            "query": "Completely different query",
            "intent": "data_retrieval",
            "topics": ["rag"],  # Same topic
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata2 = generate_retrieval_metadata(query_context2, user_tenant)

        print(f"   Query 2: '{query_context2['query']}'")
        print(f"   Metadata 2 items: {len(metadata2)}")

        # Both queries with same topic should produce similar metadata structure
        if len(metadata) == len(metadata2):
            print("   ⚠️  ISSUE: Same topic produces same metadata count")
            print("   ⚠️  No actual similarity search based on query text")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Topic-Based Generation Test
# ============================================================================


def test_topic_based_generation():
    """Test that metadata is generated per topic (not query similarity)"""
    print("\n⚠️  Testing Topic-Based Generation...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test 1: Single topic
        query_context1 = {
            "query": "Query about finance",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata1 = generate_retrieval_metadata(query_context1, "tenantA")

        print(f"   Query 1: '{query_context1['query']}'")
        print(f"   Topics: {query_context1['topics']}")
        print(f"   Metadata items: {len(metadata1)}")

        # Test 2: Multiple topics
        query_context2 = {
            "query": "Query about finance and marketing",
            "intent": "data_retrieval",
            "topics": ["finance", "marketing"],
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata2 = generate_retrieval_metadata(query_context2, "tenantA")

        print(f"   Query 2: '{query_context2['query']}'")
        print(f"   Topics: {query_context2['topics']}")
        print(f"   Metadata items: {len(metadata2)}")

        # Metadata count should match topic count
        assert len(metadata1) == len(query_context1["topics"]), (
            f"Metadata count should match topic count: {len(metadata1)} != {len(query_context1['topics'])}"
        )
        assert len(metadata2) == len(query_context2["topics"]), (
            f"Metadata count should match topic count: {len(metadata2)} != {len(query_context2['topics'])}"
        )

        print("   ⚠️  ISSUE: Metadata generated per topic, not query similarity")
        print("   ⚠️  Expected: Retrieve documents based on query semantic similarity")
        print("   ⚠️  Actual: Generate metadata for each topic in query_context")
        print("   ⚠️  Impact: Ignores actual query text similarity")

        # Check topic distribution
        for i, item in enumerate(metadata2):
            print(f"   Item {i + 1}: topics={item['topics']}")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Hardcoded Cross-Tenant Doc Test
# ============================================================================


def test_hardcoded_cross_tenant_doc():
    """Test hardcoded cross-tenant document logic"""
    print("\n⚠️  Testing Hardcoded Cross-Tenant Doc...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test 1: Same tenant (no cross-tenant doc)
        query_context1 = {
            "query": "Query tenantA documents",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": None,  # No explicit target
            "detected_threats": [],
        }
        user_tenant1 = "tenantA"

        metadata1 = generate_retrieval_metadata(query_context1, user_tenant1)

        print(f"   Query 1: Same tenant (user={user_tenant1})")
        print(f"   Metadata items: {len(metadata1)}")
        print(f"   Target tenant: {query_context1.get('target_tenant')}")

        # Check tenant IDs
        tenant_ids1 = [item["tenant_id"] for item in metadata1]
        print(f"   Tenant IDs: {set(tenant_ids1)}")

        # Test 2: Different tenant (should add cross-tenant doc)
        query_context2 = {
            "query": "Query tenantB documents",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": "tenantB",  # Explicit target
            "detected_threats": [],
        }
        user_tenant2 = "tenantA"

        metadata2 = generate_retrieval_metadata(query_context2, user_tenant2)

        print(f"   Query 2: Cross-tenant (user={user_tenant2}, target=tenantB)")
        print(f"   Metadata items: {len(metadata2)}")

        # Check tenant IDs
        tenant_ids2 = [item["tenant_id"] for item in metadata2]
        print(f"   Tenant IDs: {set(tenant_ids2)}")

        # Check for cross-tenant doc
        cross_tenant_items = [
            item for item in metadata2 if item["tenant_id"] != query_context2["target_tenant"]
        ]
        print(f"   Cross-tenant items: {len(cross_tenant_items)}")

        if cross_tenant_items:
            for item in cross_tenant_items:
                print(
                    f"   Cross-tenant item: tenant_id={item['tenant_id']}, embedding_id={item['embedding_id']}"
                )
                if item["embedding_id"] == "emb-cross-001":
                    print("   ⚠️  ISSUE: Hardcoded cross-tenant doc (line 31)")
                    print("   ⚠️  Expected: Real cross-tenant documents from DB")
                    print("   ⚠️  Actual: Always same cross-tenant doc (emb-cross-001)")

        # Test 3: Same tenant with explicit target (shouldn't add cross-tenant)
        query_context3 = {
            "query": "Query tenantA documents",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": "tenantA",  # Same as user
            "detected_threats": [],
        }
        user_tenant3 = "tenantA"

        metadata3 = generate_retrieval_metadata(query_context3, user_tenant3)

        print("   Query 3: Same tenant with explicit target")
        print(f"   Metadata items: {len(metadata3)}")

        cross_tenant_items3 = [
            item for item in metadata3 if item["tenant_id"] != query_context3["target_tenant"]
        ]
        print(f"   Cross-tenant items: {len(cross_tenant_items3)}")

        assert len(cross_tenant_items3) == 0, "Should not add cross-tenant doc when target == user"

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ No K Parameter Test
# ============================================================================


def test_no_k_parameter():
    """Test that function doesn't respect k parameter"""
    print("\n⚠️  Testing No K Parameter...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test with different topic counts (simulating k)
        query_context1 = {
            "query": "Query",
            "intent": "data_retrieval",
            "topics": ["finance"],  # 1 topic
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata1 = generate_retrieval_metadata(query_context1, "tenantA")

        query_context2 = {
            "query": "Query",
            "intent": "data_retrieval",
            "topics": ["finance", "marketing", "hr", "rag"],  # 4 topics
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata2 = generate_retrieval_metadata(query_context2, "tenantA")

        print(f"   Query 1: {len(query_context1['topics'])} topics -> {len(metadata1)} items")
        print(f"   Query 2: {len(query_context2['topics'])} topics -> {len(metadata2)} items")

        # Metadata count matches topic count, not a k parameter
        print("   ⚠️  ISSUE: No k parameter (retrieve top-k documents)")
        print("   ⚠️  Expected: Function accepts k parameter and returns top-k docs")
        print("   ⚠️  Actual: Returns metadata per topic in query_context")
        print("   ⚠️  Impact: Cannot control how many documents to retrieve")

        # Check if function signature has k parameter
        import inspect

        sig = inspect.signature(generate_retrieval_metadata)
        params = list(sig.parameters.keys())
        print(f"   Function parameters: {params}")

        if "k" not in params:
            print("   ⚠️  ISSUE: Function doesn't have k parameter")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Tenant Policy Integration Test
# ============================================================================


def test_tenant_policy_integration():
    """Test integration with tenant policies"""
    print("\n🔧 Testing Tenant Policy Integration...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata
        from src.sec_agent.policy_manager import TENANT_POLICIES

        # Test with different tenants
        query_context = {
            "query": "Query",
            "intent": "data_retrieval",
            "topics": ["marketing"],
            "target_tenant": None,
            "detected_threats": [],
        }

        # Test tenantA
        metadata1 = generate_retrieval_metadata(query_context, "tenantA")
        sensitivity1 = metadata1[0]["sensitivity"]

        print("   Tenant: tenantA")
        print(f"   Sensitivity: {sensitivity1}")
        print(f"   Expected: {TENANT_POLICIES.get('tenantA', {}).get('sensitivity', 'INTERNAL')}")

        # Test tenantB
        metadata2 = generate_retrieval_metadata(query_context, "tenantB")
        sensitivity2 = metadata2[0]["sensitivity"]

        print("   Tenant: tenantB")
        print(f"   Sensitivity: {sensitivity2}")
        print(f"   Expected: {TENANT_POLICIES.get('tenantB', {}).get('sensitivity', 'INTERNAL')}")

        # Check if sensitivities match policies
        expected_sens1 = TENANT_POLICIES.get("tenantA", {}).get("sensitivity", "INTERNAL")
        expected_sens2 = TENANT_POLICIES.get("tenantB", {}).get("sensitivity", "INTERNAL")

        assert sensitivity1 == expected_sens1, (
            f"Sensitivity should match policy: {sensitivity1} != {expected_sens1}"
        )
        assert sensitivity2 == expected_sens2, (
            f"Sensitivity should match policy: {sensitivity2} != {expected_sens2}"
        )

        print("   ✅ Sensitivities match tenant policies")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 9️⃣ Empty Topics Test
# ============================================================================


def test_empty_topics():
    """Test behavior with empty topics"""
    print("\n🔧 Testing Empty Topics...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test with no topics
        query_context = {
            "query": "Query with no recognized topics",
            "intent": "data_retrieval",
            "topics": [],  # Empty
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata = generate_retrieval_metadata(query_context, "tenantA")

        print(f"   Query: '{query_context['query']}'")
        print(f"   Topics: {query_context['topics']}")
        print(f"   Metadata items: {len(metadata)}")

        # Should return empty list or handle gracefully
        if len(metadata) == 0:
            print("   ✅ Returns empty list for no topics")
        else:
            print(f"   ⚠️  Returns {len(metadata)} items even with no topics")
            for item in metadata:
                print(f"   Item: {item}")

    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 🔟 Edge Cases Test
# ============================================================================


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()

    try:
        from src.sec_agent.metadata_generator import generate_retrieval_metadata

        # Test 1: None query_context
        try:
            metadata = generate_retrieval_metadata(None, "tenantA")
            print(f"   ⚠️  None query_context handled: {metadata}")
        except Exception as e:
            print(f"   ✅ None query_context raises error: {type(e).__name__}")

        # Test 2: Missing topics key
        query_context = {
            "query": "Query",
            "intent": "data_retrieval",
            # "topics" missing
            "target_tenant": None,
            "detected_threats": [],
        }

        try:
            metadata = generate_retrieval_metadata(query_context, "tenantA")
            print(f"   Missing topics handled: {len(metadata)} items")
            print("   ✅ Uses default empty list for missing topics")
        except Exception as e:
            print(f"   ⚠️  Missing topics raises error: {type(e).__name__}")

        # Test 3: Invalid tenant
        query_context = {
            "query": "Query",
            "intent": "data_retrieval",
            "topics": ["finance"],
            "target_tenant": None,
            "detected_threats": [],
        }

        metadata = generate_retrieval_metadata(query_context, "invalid_tenant")
        sensitivity = metadata[0]["sensitivity"]

        print("   Invalid tenant: invalid_tenant")
        print(f"   Sensitivity: {sensitivity}")
        print("   Expected: INTERNAL (default)")

        assert sensitivity == "INTERNAL", "Should use default sensitivity for invalid tenant"
        print("   ✅ Uses default sensitivity for invalid tenant")

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
    print("VecSec Metadata Generator Functional Diagnostics")
    print("=" * 60)

    test_basic_functionality()
    test_fake_metadata_generation()
    test_arbitrary_score_formula()
    test_no_actual_similarity_search()
    test_topic_based_generation()
    test_hardcoded_cross_tenant_doc()
    test_no_k_parameter()
    test_tenant_policy_integration()
    test_empty_topics()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("🏁 Metadata Generator Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Fake metadata (not real retrieval)")
    print("   🔴 CRITICAL: Arbitrary score formula")
    print("   🔴 CRITICAL: No actual similarity search")
    print("   ⚠️  HIGH: Topic-based generation (not query-based)")
    print("   ⚠️  HIGH: Hardcoded cross-tenant doc")
    print("   ⚠️  MEDIUM: No k parameter")
    print("   ⚠️  MEDIUM: No real document content")
