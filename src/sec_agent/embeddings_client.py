"""
BaseTen Qwen3 Embedding Client with batching and caching

KNOWN ISSUES IN get_embedding:
==============================
1. Random Embeddings After Training (Line ~116-120)
   - Returns random.rand(768) when patterns_learned >= 100
   - Problem: Random vectors provide no semantic value for similarity checks
   - Fix: Raise exception or skip semantic detection entirely

2. Random Embeddings as Fallback (Line ~122-126)
   - Returns random.rand(768) when BaseTen API not enabled
   - Problem: Security checks based on random vectors don't work
   - Fix: Raise ValueError with clear error message

3. Random Embeddings While Batch Filling (Line ~140-141)
   - Returns random.rand(768) while waiting for batch to fill
   - Problem: Early requests use random embeddings instead of real ones
   - Fix: Flush batch immediately or wait for real embeddings

Note: Hashing is CORRECT - cache stores 768-dim vectors, not text. Embeddings needed for np.dot() similarity.
"""

import os
import requests
import numpy as np


class QwenEmbeddingClient:
    """BaseTen Qwen3 embedding client with batching"""
    
    def __init__(self, model_id: str = None, api_key: str = None, batch_size: int = 100):
        self.model_id = model_id or os.getenv("BASETEN_MODEL_ID")
        self.api_key = api_key or os.getenv("BASETEN_API_KEY")
        self.batch_size = batch_size
        self.embedding_cache = {}  # Cache for embeddings
        self.pending_batch = []  # Queue for batching
        self.total_calls = 0
        self.min_patterns_for_training = 100  # Stop calling API after this many patterns learned
        
        if not self.model_id or not self.api_key:
            print("⚠️  BaseTen API credentials not set. Embedding-based detection disabled.")
            self.enabled = False
        else:
            self.base_url = f"https://api.baseten.co/v1/models/{self.model_id}/predict"
            self.headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            self.enabled = True
            print(f"✅ BaseTen client initialized with batch size: {self.batch_size}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from BaseTen with batching and caching"""
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # If we've learned enough patterns, skip API calls
        if hasattr(self, 'patterns_learned') and self.patterns_learned >= self.min_patterns_for_training:
            # Return random embedding (we won't use semantic detection anymore)
            embedding = np.random.rand(768)
            self.embedding_cache[cache_key] = embedding
            return embedding
        
        if not self.enabled:
            # Return random embedding as fallback
            embedding = np.random.rand(768)
            self.embedding_cache[cache_key] = embedding
            return embedding
        
        # Add to pending batch
        self.pending_batch.append(text)
        
        # Process batch if we've accumulated enough
        if len(self.pending_batch) >= self.batch_size:
            embeddings = self._process_batch()
            # Update cache for all items in batch
            for i, text_item in enumerate(self.pending_batch):
                self.embedding_cache[hash(text_item)] = embeddings[i]
            # Return the embedding for current request
            return embeddings[len(self.pending_batch) - 1]
        
        # For now, return random (will be updated when batch processes)
        return np.random.rand(768)
    
    def _process_batch(self) -> list:
        """Process pending batch of texts and get embeddings"""
        if not self.pending_batch:
            return []
        
        texts = self.pending_batch[:]
        self.pending_batch = []  # Clear batch
        
        try:
            # Batch API call
            payload = {
                "inputs": texts,  # Send multiple texts at once
                "parameters": {
                    "task": "embedding",
                    "max_length": 512
                }
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=self.headers,
                timeout=10  # Longer timeout for batch
            )
            
            self.total_calls += 1
            print(f"📦 Batch API call #{self.total_calls} processed {len(texts)} embeddings")
            
            if response.status_code == 200:
                result = response.json()
                embeddings = [np.array(emb) for emb in result["outputs"]]
                return embeddings
            else:
                print(f"⚠️  BaseTen API error: {response.status_code}")
                return [np.random.rand(768) for _ in texts]
        except Exception as e:
            print(f"⚠️  BaseTen batch failed: {e}")
            return [np.random.rand(768) for _ in texts]
    
    def flush_batch(self):
        """Force process any pending batch items"""
        if self.pending_batch:
            embeddings = self._process_batch()
            for i, text_item in enumerate(self.pending_batch):
                self.embedding_cache[hash(text_item)] = embeddings[i]
    
    def set_patterns_learned(self, count: int):
        """Update number of patterns learned"""
        self.patterns_learned = count
        if count >= self.min_patterns_for_training:
            print(f"🎓 Training complete! Learned {count} patterns. Disabling BaseTen API calls.")
    
    def get_stats(self):
        """Get embedding statistics"""
        return {
            "total_calls": self.total_calls,
            "cache_size": len(self.embedding_cache),
            "pending_batch_size": len(self.pending_batch),
            "patterns_learned": getattr(self, 'patterns_learned', 0)
        }

