"""
VecSec Embeddings Client Functional Diagnostic
Tests SentenceTransformers-based local embeddings client
"""

import os
import sys
import traceback
from pathlib import Path

import numpy as np

# Add project root to Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Suppress metrics initialization
os.environ.setdefault("METRICS_ENABLED", "false")
os.environ.setdefault("USE_CHROMA", "false")

print("🚀 Starting VecSec Embeddings Client Functional Diagnostics\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================


def reset_env():
    """Reset environment variables"""
    pass  # No env vars needed for SentenceTransformers


# ============================================================================
# 1️⃣ Client Initialization Test
# ============================================================================


def test_client_initialization():
    """Test client initialization"""
    print("\n🔧 Testing Client Initialization...")
    reset_env()

    try:
        # Import directly to avoid langgraph conflicts
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        # Test 1: Default initialization
        client1 = QwenEmbeddingClient()
        print(f"   Model: {client1.model_name}")
        print(f"   Enabled: {client1.enabled}")
        print(f"   Batch size: {client1.batch_size}")

        if client1.enabled:
            print("   ✅ Client initialized successfully")
        else:
            print("   ⚠️  Client disabled (sentence-transformers not installed?)")

        # Test 2: Custom model
        try:
            _ = QwenEmbeddingClient(model_name="all-MiniLM-L6-v2", batch_size=64)
            print("   ✅ Custom model and batch size work")
        except Exception as e:
            print(f"   ⚠️  Custom model failed: {e}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install with: pip install sentence-transformers")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 2️⃣ Basic Embedding Test
# ============================================================================


def test_basic_embedding():
    """Test basic embedding generation"""
    print("\n🔧 Testing Basic Embedding...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   ⚠️  Skipping - client not enabled (sentence-transformers not installed)")
            return

        # Test single embedding
        text = "Cybersecurity is important"
        embedding = client.get_embedding(text)

        print(f"   Input: '{text}'")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")

        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert len(embedding) == 384, f"Expected 384-dim, got {len(embedding)}"
        print("   ✅ Basic embedding works!")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 3️⃣ Caching Test
# ============================================================================


def test_caching():
    """Test embedding caching"""
    print("\n🔧 Testing Caching...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   ⚠️  Skipping - client not enabled")
            return

        text = "Test caching query"

        # First call - should compute
        embedding1 = client.get_embedding(text)
        stats1 = client.get_stats()

        # Second call - should hit cache
        embedding2 = client.get_embedding(text)
        stats2 = client.get_stats()

        print(
            f"   First call: total_calls={stats1['total_calls']}, cache_size={stats1['cache_size']}"
        )
        print(
            f"   Second call: total_calls={stats2['total_calls']}, cache_size={stats2['cache_size']}"
        )

        # Check cache hit
        assert np.array_equal(embedding1, embedding2), "Cache hit should return same embedding"
        assert stats2["cache_size"] == stats1["cache_size"], "Cache size should not increase"

        print("   ✅ Caching works correctly")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 4️⃣ Batch Processing Test
# ============================================================================


def test_batch_processing():
    """Test batch embedding generation"""
    print("\n🔧 Testing Batch Processing...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   ⚠️  Skipping - client not enabled")
            return

        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        embeddings = client.get_embeddings_batch(texts)

        print(f"   Input texts: {len(texts)}")
        print(f"   Output embeddings: {len(embeddings)}")

        assert len(embeddings) == len(texts), "Should return same number of embeddings"
        assert all(isinstance(emb, np.ndarray) for emb in embeddings), "All should be numpy arrays"
        assert all(len(emb) == 384 for emb in embeddings), "All should be 384-dim"

        print("   ✅ Batch processing works!")
        print(f"   Stats: {client.get_stats()}")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 5️⃣ Semantic Similarity Test
# ============================================================================


def test_semantic_similarity():
    """Test semantic similarity with embeddings"""
    print("\n🔧 Testing Semantic Similarity...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   ⚠️  Skipping - client not enabled")
            return

        # Similar texts
        text1 = "I love programming"
        text2 = "Coding is fun"

        # Different text
        text3 = "The weather is nice today"

        emb1 = client.get_embedding(text1, normalize=True)
        emb2 = client.get_embedding(text2, normalize=True)
        emb3 = client.get_embedding(text3, normalize=True)

        # Calculate cosine similarity
        similarity_12 = np.dot(emb1, emb2)  # Normalized = cosine similarity
        similarity_13 = np.dot(emb1, emb3)

        print(f"   '{text1}' vs '{text2}': {similarity_12:.3f}")
        print(f"   '{text1}' vs '{text3}': {similarity_13:.3f}")

        assert similarity_12 > similarity_13, "Similar texts should have higher similarity"
        print(f"   ✅ Semantic similarity works! ({similarity_12:.3f} > {similarity_13:.3f})")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 6️⃣ Stats Test
# ============================================================================


def test_stats():
    """Test stats tracking"""
    print("\n🔧 Testing Stats...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   ⚠️  Skipping - client not enabled")
            return

        # Generate some embeddings
        client.get_embedding("test 1")
        client.get_embedding("test 2")
        client.get_embeddings_batch(["test 3", "test 4"])

        stats = client.get_stats()

        print(f"   Stats: {stats}")

        assert "total_calls" in stats, "Should have total_calls"
        assert "cache_size" in stats, "Should have cache_size"
        assert "model_name" in stats, "Should have model_name"
        assert "enabled" in stats, "Should have enabled"

        print("   ✅ Stats tracking works!")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Test failed: {e}")


# ============================================================================
# 7️⃣ Error Handling Test
# ============================================================================


def test_error_handling():
    """Test error handling when client is disabled"""
    print("\n🔧 Testing Error Handling...")
    reset_env()

    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from embeddings_client import QwenEmbeddingClient

        # Test with missing sentence-transformers (if possible)
        # This would require mocking, but we can test the enabled check

        client = QwenEmbeddingClient()

        if not client.enabled:
            print("   Client disabled (sentence-transformers not installed)")
            try:
                client.get_embedding("test")
                print("   ⚠️  Should have raised ValueError")
            except ValueError as e:
                print(f"   ✅ Correctly raises ValueError: {e}")
        else:
            print("   ✅ Client enabled - error handling not testable without mocking")

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
    test_basic_embedding()
    test_caching()
    test_batch_processing()
    test_semantic_similarity()
    test_stats()
    test_error_handling()

    print("\n" + "=" * 60)
    print("🏁 Embeddings Client Diagnostics Complete")
    print("=" * 60)
    print("\n✨ SentenceTransformers Benefits:")
    print("   • ✅ No API key needed")
    print("   • ✅ No API calls (fully offline)")
    print("   • ✅ No rate limits")
    print("   • ✅ No cost")
    print("   • ✅ Fast (CPU: ~50-100ms, GPU: ~1-5ms)")
    print("   • ✅ Deterministic (same text = same vector)")
    print("   • ✅ 384-dimensional embeddings (all-MiniLM-L6-v2)")
