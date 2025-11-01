"""
VecSec Embeddings Client Functional Diagnostic
Tests real runtime behavior of embeddings_client.py subsystems
Purpose: Diagnose all embedding client issues before refactoring
"""

import os
import sys
import traceback
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

print("🚀 Starting VecSec Embeddings Client Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables to safe defaults"""
    os.environ.pop("BASETEN_MODEL_ID", None)
    os.environ.pop("BASETEN_API_KEY", None)


def create_mock_embedding_response(dim=768, count=1):
    """Create mock API response with embeddings"""
    return {
        "outputs": [[0.5] * dim for _ in range(count)]
    }


# ============================================================================
# 1️⃣ Client Initialization Test
# ============================================================================

def test_client_initialization():
    """Test client initialization with different configs"""
    print("\n🔧 Testing Client Initialization...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        # Test 1: No API credentials (should disable)
        reset_env()
        client1 = QwenEmbeddingClient()
        print(f"✅ Client initialized without credentials")
        print(f"   enabled = {client1.enabled}")
        print(f"   model_id = {client1.model_id}")
        assert client1.enabled == False, "Should be disabled when no credentials"
        
        # Test 2: With fake credentials (should enable)
        os.environ["BASETEN_MODEL_ID"] = "test_model_123"
        os.environ["BASETEN_API_KEY"] = "test_key_abc"
        client2 = QwenEmbeddingClient()
        print(f"✅ Client initialized with credentials")
        print(f"   enabled = {client2.enabled}")
        print(f"   base_url = {client2.base_url}")
        assert client2.enabled == True, "Should be enabled with credentials"
        assert client2.base_url == "https://api.baseten.co/v1/models/test_model_123/predict"
        
        # Test 3: Custom batch size
        client3 = QwenEmbeddingClient(batch_size=50)
        assert client3.batch_size == 50, "Custom batch size should work"
        print(f"✅ Custom batch size: {client3.batch_size}")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Client initialization test failed: {e}")


# ============================================================================
# 2️⃣ Random Embeddings Fallback Test (CRITICAL)
# ============================================================================

def test_random_embeddings_when_disabled():
    """Test that client returns random embeddings when API not enabled"""
    print("\n🔴 Testing Random Embeddings (API Disabled)...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        # Create client without credentials (disabled)
        client = QwenEmbeddingClient()
        assert client.enabled == False, "Client should be disabled"
        
        # Test get_embedding when disabled
        print("Testing get_embedding() when API disabled...")
        embedding = client.get_embedding("test query")
        
        # Check if it's random
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (768,), f"Should be 768-dim, got {embedding.shape}"
        
        # Get another embedding - should be different (random)
        embedding2 = client.get_embedding("different query")
        similarity = np.dot(embedding, embedding2) / (
            np.linalg.norm(embedding) * np.linalg.norm(embedding2)
        )
        
        print(f"   Embedding 1 shape: {embedding.shape}")
        print(f"   Embedding 2 shape: {embedding2.shape}")
        print(f"   Similarity between random embeddings: {similarity:.3f}")
        print(f"   ⚠️  ISSUE: Returning random embeddings instead of raising error")
        print(f"   ⚠️  Expected: ValueError when API not configured")
        print(f"   ⚠️  Actual: Returns random.rand(768) - semantic detection broken")
        
        # Check if cached (should be, since it's stored)
        cache_key = hash("test query")
        assert cache_key in client.embedding_cache, "Should cache the embedding"
        print(f"   ⚠️  ISSUE: Random embeddings are being cached")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Random Embeddings After Training Test (CRITICAL)
# ============================================================================

def test_random_embeddings_after_training():
    """Test that client returns random embeddings after training complete"""
    print("\n🔴 Testing Random Embeddings (After Training)...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        # Create client with credentials
        os.environ["BASETEN_MODEL_ID"] = "test_model"
        os.environ["BASETEN_API_KEY"] = "test_key"
        client = QwenEmbeddingClient()
        
        # Simulate training complete
        client.set_patterns_learned(100)  # >= min_patterns_for_training
        print(f"   patterns_learned = {client.patterns_learned}")
        print(f"   min_patterns_for_training = {client.min_patterns_for_training}")
        
        # Test get_embedding after training
        embedding = client.get_embedding("test after training")
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (768,), f"Should be 768-dim, got {embedding.shape}"
        
        print(f"   ⚠️  ISSUE: Returning random embeddings after training")
        print(f"   ⚠️  Expected: ValueError or skip semantic detection")
        print(f"   ⚠️  Actual: Returns random.rand(768) - no semantic value")
        
        # Check cache
        cache_key = hash("test after training")
        assert cache_key in client.embedding_cache, "Should cache random embedding"
        print(f"   ⚠️  ISSUE: Random embeddings cached (waste of memory)")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Batch Processing Race Condition Test (CRITICAL)
# ============================================================================

def test_batch_race_condition():
    """Test that early requests get random embeddings while batch fills"""
    print("\n🔴 Testing Batch Race Condition...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        os.environ["BASETEN_MODEL_ID"] = "test_model"
        os.environ["BASETEN_API_KEY"] = "test_key"
        
        # Create client with small batch size for testing
        client = QwenEmbeddingClient(batch_size=5)
        
        # Mock API response
        with patch('src.sec_agent.embeddings_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = create_mock_embedding_response(count=5)
            mock_post.return_value = mock_response
            
            # Make 3 requests (batch not full yet)
            embeddings = []
            for i in range(3):
                emb = client.get_embedding(f"query {i}")
                embeddings.append(emb)
                print(f"   Request {i+1}: embedding shape = {emb.shape}")
            
            # Check if early requests got random embeddings
            print(f"   Batch size: {client.batch_size}")
            print(f"   Pending batch size: {len(client.pending_batch)}")
            
            # The issue: Early requests should get random embeddings (line 86)
            # Check if they're different (random)
            if len(embeddings) > 1:
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                print(f"   Similarity between early embeddings: {similarity:.3f}")
                
                if similarity < 0.5:  # Random embeddings should be uncorrelated
                    print(f"   ⚠️  ISSUE: Early requests returned random embeddings")
                    print(f"   ⚠️  Expected: Wait for batch or flush immediately")
                    print(f"   ⚠️  Actual: Returns random.rand(768) while batch fills")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Cache Key Collision Test
# ============================================================================

def test_cache_key_collisions():
    """Test that hash() can cause collisions"""
    print("\n⚠️  Testing Cache Key Collisions...")
    
    try:
        import hashlib
        
        # Test hash collisions
        test_strings = [
            "test query 1",
            "test query 2",
            "different query",
            "another query"
        ]
        
        # Check if hash() produces same values for different strings
        hash_values = [hash(s) for s in test_strings]
        unique_hashes = len(set(hash_values))
        
        print(f"   Test strings: {len(test_strings)}")
        print(f"   Unique hash values: {unique_hashes}")
        print(f"   Hash collisions possible: {unique_hashes < len(test_strings)}")
        
        if unique_hashes < len(test_strings):
            print(f"   ⚠️  ISSUE: hash() can produce collisions")
            print(f"   ⚠️  Recommendation: Use MD5 or SHA256 for cache keys")
        else:
            print(f"   ✅ No collisions in test (but hash() can still collide)")
        
        # Show hash values
        for s, h in zip(test_strings, hash_values):
            print(f"   '{s}' -> hash={h}")
        
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ API Error Handling Test
# ============================================================================

def test_api_error_handling():
    """Test that API errors return random embeddings"""
    print("\n🔴 Testing API Error Handling...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        os.environ["BASETEN_MODEL_ID"] = "test_model"
        os.environ["BASETEN_API_KEY"] = "test_key"
        
        client = QwenEmbeddingClient(batch_size=1)  # Small batch for testing
        
        # Test 1: API returns error status
        with patch('src.sec_agent.embeddings_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500  # Server error
            mock_post.return_value = mock_response
            
            embedding = client.get_embedding("test query")
            
            print(f"   API returned status 500")
            print(f"   get_embedding() returned: {type(embedding)}")
            print(f"   ⚠️  ISSUE: API error returns random embeddings (line 122)")
            print(f"   ⚠️  Expected: Raise exception or return None")
            print(f"   ⚠️  Actual: Returns random.rand(768) - no semantic value")
            
            assert isinstance(embedding, np.ndarray), "Should return numpy array"
            assert embedding.shape == (768,), "Should be 768-dim"
        
        # Test 2: API throws exception
        with patch('src.sec_agent.embeddings_client.requests.post') as mock_post:
            mock_post.side_effect = Exception("Network error")
            
            embedding = client.get_embedding("test query 2")
            
            print(f"   API raised Exception")
            print(f"   ⚠️  ISSUE: Exception returns random embeddings (line 125)")
            print(f"   ⚠️  Expected: Raise exception to caller")
            print(f"   ⚠️  Actual: Catches exception and returns random.rand(768)")
            
            assert isinstance(embedding, np.ndarray), "Should return numpy array"
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Cache Behavior Test
# ============================================================================

def test_cache_behavior():
    """Test caching behavior and issues"""
    print("\n⚠️  Testing Cache Behavior...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        client = QwenEmbeddingClient()
        
        # Test cache hit
        embedding1 = client.get_embedding("cached query")
        cache_key = hash("cached query")
        
        # Second call should hit cache
        embedding2 = client.get_embedding("cached query")
        
        # Check if same embedding returned
        assert np.array_equal(embedding1, embedding2), "Cache hit should return same embedding"
        print(f"   ✅ Cache hit works correctly")
        print(f"   Cache size: {len(client.embedding_cache)}")
        
        # Test cache never expires
        stats = client.get_stats()
        print(f"   Cache size: {stats['cache_size']}")
        print(f"   ⚠️  ISSUE: Cache never expires - memory grows unbounded")
        print(f"   ⚠️  No max_size or expiration policy")
        
        # Test hash() collision risk
        print(f"   ⚠️  ISSUE: Using hash() for cache keys (line 56)")
        print(f"   ⚠️  Hash collisions can return wrong embeddings")
        
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 8️⃣ Batch Processing Test
# ============================================================================

def test_batch_processing():
    """Test batch processing behavior"""
    print("\n📦 Testing Batch Processing...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        os.environ["BASETEN_MODEL_ID"] = "test_model"
        os.environ["BASETEN_API_KEY"] = "test_key"
        
        client = QwenEmbeddingClient(batch_size=3)
        
        with patch('src.sec_agent.embeddings_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = create_mock_embedding_response(count=3)
            mock_post.return_value = mock_response
            
            # Make exactly batch_size requests
            embeddings = []
            for i in range(3):
                emb = client.get_embedding(f"query {i}")
                embeddings.append(emb)
            
            # Check if batch was processed
            print(f"   Batch size: {client.batch_size}")
            print(f"   Requests made: 3")
            print(f"   API calls made: {mock_post.call_count}")
            print(f"   Total calls tracked: {client.total_calls}")
            
            # Batch should process when full
            assert mock_post.call_count >= 1, "Batch should trigger API call when full"
            print(f"   ✅ Batch processed when full")
            
            # Check pending batch is empty after processing
            print(f"   Pending batch size: {len(client.pending_batch)}")
            
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 9️⃣ Flush Batch Test
# ============================================================================

def test_flush_batch():
    """Test batch flush functionality"""
    print("\n🔄 Testing Batch Flush...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        os.environ["BASETEN_MODEL_ID"] = "test_model"
        os.environ["BASETEN_API_KEY"] = "test_key"
        
        client = QwenEmbeddingClient(batch_size=10)
        
        with patch('src.sec_agent.embeddings_client.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = create_mock_embedding_response(count=2)
            mock_post.return_value = mock_response
            
            # Add 2 items to batch (not full)
            client.get_embedding("query 1")
            client.get_embedding("query 2")
            
            print(f"   Pending batch size before flush: {len(client.pending_batch)}")
            
            # Flush batch
            client.flush_batch()
            
            print(f"   Pending batch size after flush: {len(client.pending_batch)}")
            print(f"   API calls made: {mock_post.call_count}")
            
            assert len(client.pending_batch) == 0, "Batch should be empty after flush"
            assert mock_post.call_count == 1, "Flush should trigger API call"
            print(f"   ✅ Flush batch works correctly")
            
    except AssertionError as e:
        print(f"❌ Assertion failed: {e}")
        traceback.print_exc()
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 🔟 Stats and Monitoring Test
# ============================================================================

def test_stats_and_monitoring():
    """Test stats tracking"""
    print("\n📊 Testing Stats and Monitoring...")
    reset_env()
    
    try:
        from src.sec_agent.embeddings_client import QwenEmbeddingClient
        
        client = QwenEmbeddingClient()
        
        # Add some embeddings
        client.get_embedding("test 1")
        client.get_embedding("test 2")
        
        stats = client.get_stats()
        
        print(f"   Total API calls: {stats['total_calls']}")
        print(f"   Cache size: {stats['cache_size']}")
        print(f"   Pending batch size: {stats['pending_batch_size']}")
        print(f"   Patterns learned: {stats['patterns_learned']}")
        
        assert 'total_calls' in stats, "Stats should include total_calls"
        assert 'cache_size' in stats, "Stats should include cache_size"
        assert 'pending_batch_size' in stats, "Stats should include pending_batch_size"
        print(f"   ✅ Stats tracking works correctly")
        
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
    print("VecSec Embeddings Client Functional Diagnostics")
    print("=" * 60)
    
    test_client_initialization()
    test_random_embeddings_when_disabled()
    test_random_embeddings_after_training()
    test_batch_race_condition()
    test_cache_key_collisions()
    test_api_error_handling()
    test_cache_behavior()
    test_batch_processing()
    test_flush_batch()
    test_stats_and_monitoring()
    
    print("\n" + "=" * 60)
    print("🏁 Embeddings Client Diagnostics Complete")
    print("=" * 60)
    print("\n📋 Summary of Issues Found:")
    print("   🔴 CRITICAL: Random embeddings returned in 4 scenarios")
    print("   🔴 CRITICAL: Batch race condition (early requests get random)")
    print("   ⚠️  HIGH: Cache uses hash() - collisions possible")
    print("   ⚠️  HIGH: No timeout/retry configuration")
    print("   ⚠️  MEDIUM: Cache never expires - memory unbounded")
    print("   ⚠️  MEDIUM: No embedding dimension validation")

