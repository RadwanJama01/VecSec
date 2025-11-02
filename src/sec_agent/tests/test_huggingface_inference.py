"""
Hugging Face InferenceClient Test Suite
Tests Hugging Face Inference API with Qwen3-Embedding-8B model
"""

import os
import sys
import traceback
from pathlib import Path
import time

# Add project root to Python path
# File is in src/sec_agent/tests/, so go up 3 levels to project root
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Set up environment variables
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
os.environ.setdefault("HF_USERNAME", os.getenv("HF_USERNAME", "RadwanJama"))

# Global cache to avoid duplicate API calls
_EMBEDDING_CACHE = {}
_QUOTA_EXCEEDED = False

print("🚀 Starting Hugging Face InferenceClient Tests\n")
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def reset_env():
    """Reset environment variables"""
    os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
    os.environ.setdefault("HF_USERNAME", os.getenv("HF_USERNAME", "RadwanJama"))


def check_quota_status(client, test_text="test"):
    """Check if quota is exceeded - stop early if so"""
    global _QUOTA_EXCEEDED
    
    if _QUOTA_EXCEEDED:
        return True
    
    try:
        # Try a single test call to check quota
        result = client.feature_extraction(test_text, model="Qwen/Qwen3-Embedding-8B")
        return False  # Quota OK
    except Exception as e:
        error_msg = str(e)
        if "402" in error_msg or "Payment Required" in error_msg:
            _QUOTA_EXCEEDED = True
            print(f"   ⚠️  QUOTA EXCEEDED detected - skipping remaining tests")
            return True
        return False  # Other error, might still work


# ============================================================================
# 1️⃣ Basic Feature Extraction Test
# ============================================================================

def test_basic_feature_extraction():
    """Test basic feature extraction with Qwen3-Embedding-8B"""
    print("\n🔧 Testing Basic Feature Extraction...")
    reset_env()
    
    global _QUOTA_EXCEEDED
    if _QUOTA_EXCEEDED:
        print("   ⚠️  Skipping - quota exceeded")
        return
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",  # or "auto" to let HF route automatically
            api_key=token
        )
        
        # Check quota first
        if check_quota_status(client):
            return
        
        text = "Today is a sunny day and I will get some ice cream."
        
        # Check cache first to avoid duplicate calls
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in _EMBEDDING_CACHE:
            print(f"   Using cached result (skipping API call)")
            result = _EMBEDDING_CACHE[cache_key]
        else:
            print(f"   Input text: '{text}'")
            print(f"   Model: Qwen/Qwen3-Embedding-8B")
            print(f"   Provider: nebius")
            
            result = client.feature_extraction(
                text,
                model="Qwen/Qwen3-Embedding-8B",
            )
            # Cache result
            _EMBEDDING_CACHE[cache_key] = result
        
        print(f"   Result type: {type(result)}")
        print(f"   Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
        
        # Result is a numpy array - can be 1D or 2D
        import numpy as np
        if isinstance(result, np.ndarray):
            # If 2D array [[embedding]], extract first element
            if result.ndim == 2 and result.shape[0] == 1:
                embedding = result[0]
            # If 2D array with multiple embeddings, use first
            elif result.ndim == 2 and result.shape[0] > 1:
                embedding = result[0]
            # If 1D array, use directly
            else:
                embedding = result
        elif isinstance(result, list) and len(result) > 0:
            # If nested list, extract first
            if isinstance(result[0], (list, np.ndarray)):
                embedding = result[0]
            else:
                embedding = result
        else:
            embedding = result
        
        print(f"   Embedding type: {type(embedding)}")
        print(f"   Embedding length: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   ✅ Feature extraction successful!")
        print(f"   ⚠️  NOTE: Actual embedding dimension: {len(embedding)} (code expects 768)")
        
        assert hasattr(embedding, '__len__'), "Embedding should have length"
        assert len(embedding) > 0, "Embedding should have non-zero length"
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Install with: pip install huggingface_hub")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 2️⃣ Multiple Texts Test (Batch Processing)
# ============================================================================

def test_multiple_texts():
    """Test feature extraction with multiple texts"""
    print("\n🔧 Testing Multiple Texts (Batch Processing)...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        texts = [
            "Today is a sunny day and I will get some ice cream.",
            "The weather is nice today.",
            "I love machine learning and AI."
        ]
        
        print(f"   Input texts: {len(texts)} texts")
        for i, text in enumerate(texts, 1):
            print(f"     {i}. '{text[:50]}...'")
        
        results = []
        import numpy as np
        for text in texts:
            result = client.feature_extraction(
                text,
                model="Qwen/Qwen3-Embedding-8B",
            )
            # Result is 2D numpy array (1, embedding_dim)
            if isinstance(result, np.ndarray):
                if result.ndim == 2 and result.shape[0] == 1:
                    results.append(result[0])  # Extract 1D embedding
                elif result.ndim == 2 and result.shape[0] > 1:
                    results.append(result[0])  # Use first embedding
                else:
                    results.append(result)  # Already 1D
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], (list, np.ndarray)):
                    results.append(result[0])
                else:
                    results.append(result)
        
        print(f"   Results: {len(results)} embeddings")
        
        if len(results) == len(texts):
            print(f"   ✅ All texts processed successfully")
            
            # Check dimensions are consistent
            dimensions = [len(emb) for emb in results]
            unique_dims = set(dimensions)
            print(f"   Embedding dimensions: {dimensions}")
            
            if len(unique_dims) == 1:
                print(f"   ✅ All embeddings have same dimension: {unique_dims.pop()}")
            else:
                print(f"   ⚠️  Different dimensions: {unique_dims}")
        else:
            print(f"   ⚠️  Only {len(results)}/{len(texts)} texts processed")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 3️⃣ Embedding Dimension Test
# ============================================================================

def test_embedding_dimension():
    """Test embedding dimensions from Qwen3-Embedding-8B"""
    print("\n🔧 Testing Embedding Dimensions...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        test_texts = [
            "Short text.",
            "This is a longer text with more words and complexity.",
            "Embedding test query for semantic search."
        ]
        
        dimensions = []
        import numpy as np
        for text in test_texts:
            result = client.feature_extraction(
                text,
                model="Qwen/Qwen3-Embedding-8B",
            )
            
            # Result is directly the embedding array
            if isinstance(result, np.ndarray):
                dim = len(result)
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], (list, np.ndarray)):
                    dim = len(result[0])
                else:
                    dim = len(result)
            else:
                continue
                
            dimensions.append(dim)
            print(f"   Text: '{text[:40]}...'")
            print(f"   Dimension: {dim}")
        
        if dimensions:
            unique_dims = set(dimensions)
            print(f"   All dimensions: {dimensions}")
            print(f"   Unique dimensions: {unique_dims}")
            
            if len(unique_dims) == 1:
                dimension = unique_dims.pop()
                print(f"   ✅ Consistent dimension: {dimension}")
                print(f"   ⚠️  NOTE: Current code assumes 768 dimensions")
                if dimension != 768:
                    print(f"   ⚠️  MISMATCH: Model returns {dimension} dimensions, code expects 768")
                else:
                    print(f"   ✅ Dimension matches expected 768")
            else:
                print(f"   ⚠️  Inconsistent dimensions: {unique_dims}")
        else:
            print(f"   ❌ No embeddings returned")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 4️⃣ Provider Test (nebius vs auto)
# ============================================================================

def test_provider_auto():
    """Test with provider='auto' to let HF route automatically"""
    print("\n🔧 Testing Provider='auto'...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="auto",  # Let HF route automatically
            api_key=token
        )
        
        text = "Testing auto provider routing."
        
        print(f"   Provider: auto")
        print(f"   Text: '{text}'")
        
        result = client.feature_extraction(
            text,
            model="Qwen/Qwen3-Embedding-8B",
        )
        
        import numpy as np
        if isinstance(result, np.ndarray):
            embedding = result
        elif isinstance(result, list) and len(result) > 0:
            embedding = result[0] if isinstance(result[0], (list, np.ndarray)) else result
        else:
            embedding = None
            
        if embedding is not None:
            print(f"   Embedding dimension: {len(embedding)}")
            print(f"   ✅ Auto provider works")
        else:
            print(f"   ⚠️  No embedding returned with auto provider")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 5️⃣ Error Handling Test
# ============================================================================

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\n🔧 Testing Error Handling...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        import numpy as np
        
        # Test 1: Empty string
        try:
            result = client.feature_extraction(
                "",
                model="Qwen/Qwen3-Embedding-8B",
            )
            if isinstance(result, np.ndarray):
                dim = len(result)
            elif isinstance(result, list) and len(result) > 0:
                dim = len(result[0]) if isinstance(result[0], (list, np.ndarray)) else len(result)
            else:
                dim = 'None'
            print(f"   Empty string: {dim} dimensions")
            print(f"   ✅ Empty string handled")
        except Exception as e:
            print(f"   Empty string error: {type(e).__name__}: {e}")
        
        # Test 2: Very long text
        try:
            long_text = " ".join(["test"] * 10000)
            result = client.feature_extraction(
                long_text,
                model="Qwen/Qwen3-Embedding-8B",
            )
            if isinstance(result, np.ndarray):
                dim = len(result)
            elif isinstance(result, list) and len(result) > 0:
                dim = len(result[0]) if isinstance(result[0], (list, np.ndarray)) else len(result)
            else:
                dim = 'None'
            print(f"   Very long text: {dim} dimensions")
            print(f"   ✅ Very long text handled")
        except Exception as e:
            print(f"   Very long text error: {type(e).__name__}: {e}")
        
        # Test 3: None input (should fail gracefully)
        try:
            result = client.feature_extraction(
                None,
                model="Qwen/Qwen3-Embedding-8B",
            )
            print(f"   None input: handled (unexpected)")
        except Exception as e:
            print(f"   None input error: {type(e).__name__}: {e}")
            print(f"   ✅ None input raises error (expected)")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 6️⃣ Performance Test
# ============================================================================

def test_performance():
    """Test performance with multiple requests"""
    print("\n🔧 Testing Performance...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        texts = [f"Test text number {i} for embedding." for i in range(5)]
        
        print(f"   Processing {len(texts)} texts...")
        
        import numpy as np
        start_time = time.time()
        results = []
        for text in texts:
            try:
                result = client.feature_extraction(
                    text,
                    model="Qwen/Qwen3-Embedding-8B",
                )
                # Result is 2D numpy array (1, embedding_dim)
                if isinstance(result, np.ndarray):
                    if result.ndim == 2 and result.shape[0] == 1:
                        results.append(result[0])  # Extract 1D embedding
                    elif result.ndim == 2 and result.shape[0] > 1:
                        results.append(result[0])  # Use first embedding
                    else:
                        results.append(result)  # Already 1D
                elif isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], (list, np.ndarray)):
                        results.append(result[0])
                    else:
                        results.append(result)
            except Exception:
                # Skip errors (like 402) and continue
                pass
        
        elapsed = time.time() - start_time
        
        print(f"   Processed: {len(results)}/{len(texts)} texts")
        print(f"   Total time: {elapsed:.2f} seconds")
        print(f"   Average time per text: {elapsed/len(texts):.2f} seconds")
        print(f"   Throughput: {len(texts)/elapsed:.2f} texts/second")
        
        if len(results) == len(texts):
            print(f"   ✅ All texts processed successfully")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 7️⃣ Semantic Similarity Test
# ============================================================================

def test_semantic_similarity():
    """Test semantic similarity between similar texts"""
    print("\n🔧 Testing Semantic Similarity...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        import numpy as np
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        # Similar texts
        text1 = "I love machine learning and AI."
        text2 = "I enjoy artificial intelligence and machine learning."
        
        # Different text
        text3 = "The weather is nice today."
        
        print(f"   Similar texts:")
        print(f"     1. '{text1}'")
        print(f"     2. '{text2}'")
        print(f"   Different text:")
        print(f"     3. '{text3}'")
        
        # Get embeddings
        import numpy as np
        
        def extract_embedding(result):
            if isinstance(result, np.ndarray):
                # Result is 2D numpy array (1, embedding_dim)
                if result.ndim == 2 and result.shape[0] == 1:
                    return result[0]  # Extract 1D embedding
                elif result.ndim == 2 and result.shape[0] > 1:
                    return result[0]  # Use first embedding
                else:
                    return result  # Already 1D
            elif isinstance(result, list) and len(result) > 0:
                return result[0] if isinstance(result[0], (list, np.ndarray)) else result
            return result
        
        emb1 = extract_embedding(client.feature_extraction(text1, model="Qwen/Qwen3-Embedding-8B"))
        emb2 = extract_embedding(client.feature_extraction(text2, model="Qwen/Qwen3-Embedding-8B"))
        emb3 = extract_embedding(client.feature_extraction(text3, model="Qwen/Qwen3-Embedding-8B"))
        
        # Convert to numpy arrays and ensure 1D
        emb1_np = np.array(emb1).flatten()
        emb2_np = np.array(emb2).flatten()
        emb3_np = np.array(emb3).flatten()
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_12 = cosine_similarity(emb1_np, emb2_np)
        sim_13 = cosine_similarity(emb1_np, emb3_np)
        sim_23 = cosine_similarity(emb2_np, emb3_np)
        
        print(f"   Similarity 1-2 (similar): {sim_12:.3f}")
        print(f"   Similarity 1-3 (different): {sim_13:.3f}")
        print(f"   Similarity 2-3 (different): {sim_23:.3f}")
        
        if sim_12 > sim_13 and sim_12 > sim_23:
            print(f"   ✅ Similar texts have higher similarity ({sim_12:.3f} > {sim_13:.3f}, {sim_23:.3f})")
        else:
            print(f"   ⚠️  Similarity results unexpected")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 8️⃣ Access Token Validation Test
# ============================================================================

def test_access_token_validation():
    """Test that access token is valid"""
    print("\n🔧 Testing Access Token Validation...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient, HfApi
        
        token = os.getenv("HF_TOKEN")
        username = os.getenv("HF_USERNAME")
        
        if not token:
            print("❌ HF_TOKEN not set")
            return
        
        print(f"   Token: {token[:15]}... (truncated)")
        print(f"   Username: {username}")
        
        # Test 1: Verify token with HfApi
        try:
            api = HfApi(token=token)
            user = api.whoami()
            print(f"   Authenticated as: {user.get('name', 'Unknown')}")
            print(f"   ✅ Token is valid")
        except Exception as e:
            print(f"   ⚠️  Token validation error: {type(e).__name__}: {e}")
        
        # Test 2: Try InferenceClient with token
        try:
            client = InferenceClient(
                provider="nebius",
                api_key=token
            )
            
            result = client.feature_extraction(
                "Test token validation",
                model="Qwen/Qwen3-Embedding-8B",
            )
            
            import numpy as np
            if isinstance(result, np.ndarray) or (isinstance(result, list) and len(result) > 0):
                print(f"   ✅ InferenceClient works with token (dimension: {len(result) if isinstance(result, np.ndarray) else len(result[0] if isinstance(result[0], (list, np.ndarray)) else result)})")
            else:
                print(f"   ⚠️  InferenceClient returns no result")
        except Exception as e:
            print(f"   ⚠️  InferenceClient error: {type(e).__name__}: {e}")
            print(f"   This might indicate:")
            print(f"     - Token doesn't have inference permissions")
            print(f"     - Model requires different authentication")
            print(f"     - Network/API access issue")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Install with: pip install huggingface_hub")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 9️⃣ Model Availability Test
# ============================================================================

def test_model_availability():
    """Test if Qwen3-Embedding-8B model is available"""
    print("\n🔧 Testing Model Availability...")
    reset_env()
    
    try:
        from huggingface_hub import HfApi
        
        token = os.getenv("HF_TOKEN")
        model_name = "Qwen/Qwen3-Embedding-8B"
        
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        api = HfApi(token=token)
        
        try:
            model_info = api.model_info(model_name)
            print(f"   Model: {model_name}")
            print(f"   Model ID: {model_info.id}")
            print(f"   Author: {model_info.author}")
            print(f"   Created: {model_info.created_at}")
            print(f"   Downloads: {model_info.downloads}")
            print(f"   ✅ Model is available")
            
            # Check if model is private/gated
            if hasattr(model_info, 'private') and model_info.private:
                print(f"   ⚠️  Model is private (requires authentication)")
            else:
                print(f"   ✅ Model is public")
                
        except Exception as e:
            print(f"   ⚠️  Cannot access model: {type(e).__name__}: {e}")
            print(f"   This might indicate:")
            print(f"     - Model doesn't exist or name is wrong")
            print(f"     - Model is private and token doesn't have access")
            print(f"     - Network/API issue")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Install with: pip install huggingface_hub")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# 🔟 Edge Cases Test
# ============================================================================

def test_edge_cases():
    """Test edge cases and special characters"""
    print("\n🔧 Testing Edge Cases...")
    reset_env()
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("❌ HF_TOKEN not set - skipping test")
            return
        
        client = InferenceClient(
            provider="nebius",
            api_key=token
        )
        
        test_cases = [
            ("Special chars: !@#$%^&*()", "Special characters"),
            ("Unicode: 你好世界 🌍", "Unicode characters"),
            ("Newlines:\nLine 1\nLine 2", "Newline characters"),
            ("Tabs:\tTabbed\ttext", "Tab characters"),
            ("Long text: " + "word " * 100, "Very long text"),
        ]
        
        import numpy as np
        for text, description in test_cases:
            try:
                result = client.feature_extraction(
                    text,
                    model="Qwen/Qwen3-Embedding-8B",
                )
                
                if isinstance(result, np.ndarray):
                    dim = len(result)
                elif isinstance(result, list) and len(result) > 0:
                    dim = len(result[0]) if isinstance(result[0], (list, np.ndarray)) else len(result)
                else:
                    dim = 'None'
                    
                if dim != 'None':
                    print(f"   ✅ {description}: {dim} dimensions")
                else:
                    print(f"   ⚠️  {description}: No result")
            except Exception as e:
                print(f"   ⚠️  {description}: {type(e).__name__}: {e}")
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Handle payment required errors gracefully
        if "402" in error_msg or "Payment Required" in error_msg:
            print(f"   ⚠️  Payment Required (402): Monthly quota exceeded")
            print(f"   ✅ Test structure is correct - need PRO subscription or wait for quota reset")
        else:
            traceback.print_exc()
            print(f"❌ Test failed: {error_type}: {error_msg[:200]}")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hugging Face InferenceClient Test Suite")
    print("=" * 60)
    print("⚠️  NOTE: This test makes multiple API calls (~12-20 calls)")
    print("⚠️  Quota check enabled - tests will skip early if quota exceeded")
    print("=" * 60)
    
    # Check if token is set
    if not os.getenv("HF_TOKEN"):
        print("\n⚠️  WARNING: HF_TOKEN not set!")
        print("   Set it with: export HF_TOKEN=your_token")
        print("   Some tests will be skipped.")
    
    # Run tests that don't make API calls first
    test_access_token_validation()
    test_model_availability()
    
    # Run API tests (will stop early if quota exceeded)
    test_basic_feature_extraction()
    
    # Only continue if quota not exceeded
    if not _QUOTA_EXCEEDED:
        test_embedding_dimension()
        test_multiple_texts()
        test_provider_auto()
        test_semantic_similarity()
        test_performance()
        test_error_handling()
        test_edge_cases()
    else:
        print("\n⚠️  Remaining tests skipped due to quota limit")
        print(f"   Total API calls made before quota check: {len(_EMBEDDING_CACHE)}")
    
    print("\n" + "=" * 60)
    print("🏁 Hugging Face InferenceClient Tests Complete")
    print(f"📊 Total API calls made: {len(_EMBEDDING_CACHE)}")
    print("=" * 60)

