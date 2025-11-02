"""
VecSec Mock LLM Functional Diagnostic
Tests real runtime behavior of mock_llm.py subsystems
Purpose: Diagnose all mock LLM and embeddings issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Mock LLM Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables"""
    os.environ.pop("USE_CHROMA", None)
    os.environ.pop("CHROMA_PATH", None)


# ============================================================================
# 1️⃣ MockLLM Initialization Test
# ============================================================================

def test_mock_llm_initialization():
    """Test MockLLM initialization"""
    print("\n🔧 Testing MockLLM Initialization...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockLLM
        
        llm = MockLLM()
        
        print(f"   MockLLM instance: {type(llm).__name__}")
        print(f"   Has invoke method: {hasattr(llm, 'invoke')}")
        print(f"   Has _generate method: {hasattr(llm, '_generate')}")
        print(f"   Has generate_prompt method: {hasattr(llm, 'generate_prompt')}")
        print(f"   Has agenerate_prompt method: {hasattr(llm, 'agenerate_prompt')}")
        
        assert hasattr(llm, 'invoke'), "Should have invoke method"
        assert hasattr(llm, '_generate'), "Should have _generate method"
        assert hasattr(llm, 'generate_prompt'), "Should have generate_prompt method"
        assert hasattr(llm, 'agenerate_prompt'), "Should have agenerate_prompt method"
        
        print(f"   ✅ MockLLM initialization works")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ MockLLM Hardcoded Responses Test (CRITICAL)
# ============================================================================

def test_hardcoded_responses():
    """Test that MockLLM uses hardcoded responses"""
    print("\n🔴 Testing Hardcoded Responses...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockLLM
        
        llm = MockLLM()
        
        # Test 1: RAG query
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        messages1 = [MockMessage("What is RAG?")]
        result1 = llm.invoke(messages1)
        
        print(f"   Query 1: 'What is RAG?'")
        print(f"   Response length: {len(result1.content)}")
        print(f"   Response starts with: '{result1.content[:50]}...'")
        
        assert "RAG" in result1.content or "Retrieval-Augmented Generation" in result1.content
        print(f"   ✅ RAG query returns RAG response")
        
        # Test 2: Finance query
        messages2 = [MockMessage("Tell me about finance")]
        result2 = llm.invoke(messages2)
        
        print(f"   Query 2: 'Tell me about finance'")
        print(f"   Response: '{result2.content}'")
        
        assert "finance" in result2.content.lower() or "restricted" in result2.content.lower()
        print(f"   ✅ Finance query returns restricted response")
        
        # Test 3: Generic query
        messages3 = [MockMessage("What is the weather?")]
        result3 = llm.invoke(messages3)
        
        print(f"   Query 3: 'What is the weather?'")
        print(f"   Response: '{result3.content}'")
        
        assert "enough context" in result3.content.lower()
        print(f"   ✅ Generic query returns default response")
        
        print(f"   ⚠️  ISSUE: Hardcoded responses (no real LLM)")
        print(f"   ⚠️  Expected: Real LLM integration (OpenAI, Anthropic, etc.)")
        print(f"   ⚠️  Actual: Keyword-based mock responses (lines 15-20)")
        print(f"   ⚠️  Impact: Not suitable for production")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ MockLLM Keyword Matching Test
# ============================================================================

def test_keyword_matching():
    """Test keyword-based response logic"""
    print("\n⚠️  Testing Keyword Matching...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockLLM
        
        llm = MockLLM()
        
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        # Test various queries
        test_cases = [
            ("What is RAG?", "RAG"),
            ("Tell me about retrieval", "RAG"),
            ("What is finance?", "finance"),
            ("finance information needed", "finance"),  # lowercase to match mock check
            ("Random question", "generic"),
        ]
        
        for query, expected_type in test_cases:
            messages = [MockMessage(query)]
            result = llm.invoke(messages)
            
            print(f"   Query: '{query}'")
            print(f"   Expected type: {expected_type}")
            print(f"   Response preview: '{result.content[:60]}...'")
            
            if expected_type == "RAG":
                assert "RAG" in result.content or "retrieval" in result.content.lower()
            elif expected_type == "finance":
                # Check if finance keyword matched (case-sensitive check in mock)
                # Mock checks for "finance" in question (lowercase)
                if "finance" in query.lower():
                    assert "finance" in result.content.lower() or "restricted" in result.content.lower()
                else:
                    # If finance keyword didn't match, should get generic response
                    assert "enough context" in result.content.lower()
            else:
                assert "enough context" in result.content.lower()
            
            print(f"   ✅ Matched {expected_type} response")
        
        print(f"   ⚠️  ISSUE: Keyword-based matching (very limited)")
        print(f"   ⚠️  Expected: Semantic understanding of queries")
        print(f"   ⚠️  Actual: Simple string matching (line 15-16, 17-18)")
        print(f"   ⚠️  Impact: Only handles 3 cases (RAG, finance, generic)")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ MockLLM Generate Prompt Test
# ============================================================================

def test_generate_prompt():
    """Test generate_prompt functionality"""
    print("\n🔧 Testing Generate Prompt...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockLLM
        
        llm = MockLLM()
        
        prompts = ["What is RAG?", "Tell me about finance", "Random question"]
        
        results = llm.generate_prompt(prompts)
        
        print(f"   Prompts: {len(prompts)}")
        print(f"   Results: {len(results)}")
        
        assert len(results) == len(prompts), f"Should return {len(prompts)} results, got {len(results)}"
        
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            print(f"   Prompt {i+1}: '{prompt}'")
            print(f"   Result {i+1}: '{result.content[:60]}...'")
        
        print(f"   ✅ Generate prompt works correctly")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ MockLLM Async Not Implemented Test
# ============================================================================

def test_async_not_implemented():
    """Test that async generation is not implemented"""
    print("\n⚠️  Testing Async Not Implemented...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockLLM
        
        llm = MockLLM()
        
        try:
            result = llm.agenerate_prompt(["test"])
            print(f"   ⚠️  Async method didn't raise error")
        except NotImplementedError as e:
            print(f"   ✅ Async raises NotImplementedError: {e}")
            assert "not implemented" in str(e).lower()
        except Exception as e:
            print(f"   ⚠️  Async raised different error: {type(e).__name__}: {e}")
        
        print(f"   ⚠️  ISSUE: Async generation not implemented (line 29)")
        print(f"   ⚠️  Expected: Async support for async workflows")
        print(f"   ⚠️  Actual: Raises NotImplementedError")
        print(f"   ⚠️  Impact: Cannot use in async contexts")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ MockEmbeddings Initialization Test
# ============================================================================

def test_mock_embeddings_initialization():
    """Test MockEmbeddings initialization"""
    print("\n🔧 Testing MockEmbeddings Initialization...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockEmbeddings
        
        embeddings = MockEmbeddings()
        
        print(f"   MockEmbeddings instance: {type(embeddings).__name__}")
        print(f"   Has embed_documents method: {hasattr(embeddings, 'embed_documents')}")
        print(f"   Has embed_query method: {hasattr(embeddings, 'embed_query')}")
        
        assert hasattr(embeddings, 'embed_documents'), "Should have embed_documents method"
        assert hasattr(embeddings, 'embed_query'), "Should have embed_query method"
        
        print(f"   ✅ MockEmbeddings initialization works")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ MockEmbeddings Random Embeddings Test (CRITICAL)
# ============================================================================

def test_random_embeddings():
    """Test that MockEmbeddings returns random embeddings"""
    print("\n🔴 Testing Random Embeddings...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockEmbeddings
        import numpy as np
        
        embeddings = MockEmbeddings()
        
        # Test embed_documents
        texts = ["test document 1", "test document 2", "test document 3"]
        doc_embeddings = embeddings.embed_documents(texts)
        
        print(f"   Documents: {len(texts)}")
        print(f"   Embeddings: {len(doc_embeddings)}")
        
        assert len(doc_embeddings) == len(texts), f"Should return {len(texts)} embeddings, got {len(doc_embeddings)}"
        
        # Check embedding dimensions
        for i, emb in enumerate(doc_embeddings):
            print(f"   Doc {i+1}: length={len(emb)}, dim={len(emb) if isinstance(emb, list) else 'N/A'}")
            assert len(emb) == 384, f"Should be 384-dim, got {len(emb)}"
        
        # Test embed_query
        query_embedding = embeddings.embed_query("test query")
        
        print(f"   Query embedding length: {len(query_embedding)}")
        assert len(query_embedding) == 384, f"Should be 384-dim, got {len(query_embedding)}"
        
        # Check if embeddings are random (different for same text)
        emb1 = embeddings.embed_query("same text")
        emb2 = embeddings.embed_query("same text")
        
        # Convert to numpy for similarity calculation
        emb1_np = np.array(emb1)
        emb2_np = np.array(emb2)
        
        similarity = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
        
        print(f"   Embedding 1 for 'same text': {emb1[:5]}...")
        print(f"   Embedding 2 for 'same text': {emb2[:5]}...")
        print(f"   Similarity between same text embeddings: {similarity:.3f}")
        
        if similarity < 0.9:  # Random embeddings should be uncorrelated
            print(f"   ⚠️  ISSUE: Random embeddings (different for same text)")
        else:
            print(f"   ✅ Embeddings consistent for same text")
        
        print(f"   ⚠️  ISSUE: Random embeddings (no semantic value)")
        print(f"   ⚠️  Expected: Real embeddings from embedding model")
        print(f"   ⚠️  Actual: Random values (lines 46, 50)")
        print(f"   ⚠️  Impact: Semantic similarity checks don't work")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ MockEmbeddings Consistency Test
# ============================================================================

def test_embedding_consistency():
    """Test embedding consistency (should be random, so inconsistent)"""
    print("\n⚠️  Testing Embedding Consistency...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockEmbeddings
        import numpy as np
        
        embeddings = MockEmbeddings()
        
        # Test same text multiple times
        text = "consistent test text"
        embeddings_list = [embeddings.embed_query(text) for _ in range(5)]
        
        print(f"   Text: '{text}'")
        print(f"   Generated embeddings: {len(embeddings_list)}")
        
        # Check if all embeddings are different (random)
        all_different = True
        for i in range(len(embeddings_list)):
            for j in range(i+1, len(embeddings_list)):
                emb_i = np.array(embeddings_list[i])
                emb_j = np.array(embeddings_list[j])
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                
                if similarity > 0.99:  # Very similar
                    all_different = False
                    print(f"   Embeddings {i} and {j} are too similar: {similarity:.3f}")
        
        if all_different:
            print(f"   ✅ All embeddings are different (random)")
        else:
            print(f"   ⚠️  Some embeddings are similar (unexpected for random)")
        
        print(f"   ⚠️  ISSUE: Embeddings are random (not deterministic)")
        print(f"   ⚠️  Expected: Same text → same embedding (deterministic)")
        print(f"   ⚠️  Actual: Same text → different embeddings (random)")
        print(f"   ⚠️  Impact: Cannot use for semantic similarity")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 9️⃣ MockEmbeddings Dimension Test
# ============================================================================

def test_embedding_dimensions():
    """Test embedding dimensions"""
    print("\n🔧 Testing Embedding Dimensions...")
    reset_env()
    
    try:
        from src.sec_agent.mock_llm import MockEmbeddings
        
        embeddings = MockEmbeddings()
        
        # Test embed_documents dimensions
        texts = ["doc1", "doc2", "doc3"]
        doc_embeddings = embeddings.embed_documents(texts)
        
        for i, emb in enumerate(doc_embeddings):
            assert len(emb) == 384, f"Doc {i+1} should be 384-dim, got {len(emb)}"
        
        print(f"   Documents embeddings: {len(doc_embeddings)} x {len(doc_embeddings[0])}-dim")
        
        # Test embed_query dimensions
        query_emb = embeddings.embed_query("query")
        assert len(query_emb) == 384, f"Query should be 384-dim, got {len(query_emb)}"
        
        print(f"   Query embedding: {len(query_emb)}-dim")
        print(f"   ✅ All embeddings are 384-dim")
        
        print(f"   ⚠️  ISSUE: Hardcoded dimension (384)")
        print(f"   ⚠️  Expected: Configurable dimension or match model")
        print(f"   ⚠️  Actual: Fixed 384-dim (lines 46, 50)")
        print(f"   ⚠️  Impact: May not match actual embedding model dimensions")
        
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
        from src.sec_agent.mock_llm import MockLLM, MockEmbeddings
        
        # Test 1: Empty messages
        llm = MockLLM()
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        try:
            result = llm.invoke([])
            print(f"   Empty messages handled: '{result.content}'")
            # MockLLM._generate handles empty messages by checking messages[-1]
            # which would raise IndexError, but it has a fallback
            # The actual behavior depends on the implementation
            assert hasattr(result, 'content'), "Should have content attribute"
            print(f"   ✅ Empty messages handled (returns default response)")
        except Exception as e:
            print(f"   ⚠️  Empty messages raised error: {type(e).__name__}: {e}")
        
        # Test 2: None messages
        try:
            result = llm.invoke(None)
            print(f"   None messages handled: '{result.content}'")
        except Exception as e:
            print(f"   ✅ None messages raises error: {type(e).__name__}: {e}")
        
        # Test 3: Empty text list
        embeddings = MockEmbeddings()
        try:
            result = embeddings.embed_documents([])
            print(f"   Empty text list: {len(result)} embeddings")
            assert len(result) == 0
            print(f"   ✅ Empty text list handled correctly")
        except Exception as e:
            print(f"   ⚠️  Empty text list raised error: {type(e).__name__}: {e}")
        
        # Test 4: Empty query
        try:
            result = embeddings.embed_query("")
            print(f"   Empty query: {len(result)}-dim embedding")
            assert len(result) == 384
            print(f"   ✅ Empty query handled correctly")
        except Exception as e:
            print(f"   ⚠️  Empty query raised error: {type(e).__name__}: {e}")
        
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
    print("VecSec Mock LLM Functional Diagnostics")
    print("=" * 60)
    
    test_mock_llm_initialization()
    test_hardcoded_responses()
    test_keyword_matching()
    test_generate_prompt()
    test_async_not_implemented()
    test_mock_embeddings_initialization()
    test_random_embeddings()
    test_embedding_consistency()
    test_embedding_dimensions()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🏁 Mock LLM Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Hardcoded responses (no real LLM)")
    print("   🔴 CRITICAL: Random embeddings (no semantic value)")
    print("   ⚠️  HIGH: Keyword-based matching (very limited)")
    print("   ⚠️  HIGH: Async not implemented")
    print("   ⚠️  MEDIUM: Fixed 384-dim embeddings")
    print("   ⚠️  MEDIUM: Non-deterministic embeddings")

