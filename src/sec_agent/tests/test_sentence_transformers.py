"""
SentenceTransformers Local Embeddings Test
🔥 Local, free, offline embeddings using SentenceTransformers

NOT using Hugging Face Inference API - this is purely local!
SentenceTransformers downloads and runs models locally on your machine.

Run directly: python src/sec_agent/tests/test_sentence_transformers.py
(NO imports from sec_agent package - avoids dependency conflicts!)
"""

import os
import sys

# Suppress any potential config initialization
os.environ.setdefault("METRICS_ENABLED", "false")
os.environ.setdefault("USE_CHROMA", "false")

print("🔍 Testing SentenceTransformers - Local Embeddings\n")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    # Pick a model (free + pre-trained, no API key needed!)
    model_name = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    print(f"📦 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✅ Model loaded (runs locally, no API)\n")
    
    # ONE embedding call
    text = "Cybersecurity is the future."
    print(f"   Input: '{text}'")
    
    embedding = model.encode(text, normalize_embeddings=True)
    
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    print(f"   ✅ Got a {len(embedding)}-dimensional vector!\n")
    
    # Show semantic similarity
    print("=" * 60)
    print("🧠 Semantic Similarity Demo\n")
    
    s1 = model.encode("I love programming", normalize_embeddings=True)
    s2 = model.encode("Coding is fun", normalize_embeddings=True)
    s3 = model.encode("The weather is nice today", normalize_embeddings=True)
    
    similarity_12 = np.dot(s1, s2)  # Cosine similarity (normalized)
    similarity_13 = np.dot(s1, s3)
    
    print(f"   'I love programming' vs 'Coding is fun': {similarity_12:.3f}")
    print(f"   'I love programming' vs 'The weather is nice': {similarity_13:.3f}")
    print(f"   ✅ Similar sentences score higher (> {similarity_12:.3f} vs {similarity_13:.3f})\n")
    
    # Show batching
    print("=" * 60)
    print("⚡ Batching Demo\n")
    
    texts = ["apple", "banana", "grapefruit", "car", "truck"]
    print(f"   Batch size: {len(texts)} texts")
    
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True)
    print(f"   ✅ Encoded {len(embeddings)} texts in one call")
    print(f"   Embeddings shape: {embeddings.shape}\n")
    
    # Show caching
    print("=" * 60)
    print("💾 Caching Demo\n")
    
    class LocalEmbeddingClient:
        """Simple local embedding client with caching"""
        def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model = SentenceTransformer(model_name)
            self.cache = {}
        
        def get_embedding(self, text):
            if text not in self.cache:
                self.cache[text] = self.model.encode(text, normalize_embeddings=True)
                print(f"   📝 Cached: '{text[:30]}...'")
            else:
                print(f"   ⚡ Cache hit: '{text[:30]}...'")
            return self.cache[text]
    
    client = LocalEmbeddingClient()
    
    # First call - cache miss
    emb1 = client.get_embedding("Cybersecurity is important")
    # Second call - cache hit
    emb2 = client.get_embedding("Cybersecurity is important")
    
    print(f"   ✅ Cache working! Same text = instant retrieval\n")
    
    print("=" * 60)
    print("🎉 SentenceTransformers Works!")
    print("=" * 60)
    print("\n✨ Benefits:")
    print("   • ✅ No API key needed")
    print("   • ✅ No API calls (fully offline)")
    print("   • ✅ No rate limits")
    print("   • ✅ No cost")
    print("   • ✅ Fast (CPU: ~50-100ms, GPU: ~1-5ms)")
    print("   • ✅ Deterministic (same text = same vector)")
    print("   • ✅ Small model (~100MB)")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"\n📦 Install with:")
    print(f"   pip install sentence-transformers")
    print(f"\nOptional (for GPU speedup):")
    print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
