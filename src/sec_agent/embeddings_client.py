"""
SentenceTransformers Local Embedding Client
🔥 Local, free, offline embeddings using SentenceTransformers

No API needed! Uses pre-trained models that run locally.
"""

import os
import numpy as np
from typing import Optional

# Try to import SentenceTransformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class EmbeddingClient:
    """
    Local embedding client using SentenceTransformers
    
    Uses pre-trained models (like 'all-MiniLM-L6-v2') that run locally.
    No API keys, no API calls, fully offline!
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize local embedding client
        
        Args:
            model_name: SentenceTransformer model name (default: all-MiniLM-L6-v2)
            batch_size: Batch size for encoding (default: 32)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.embedding_cache = {}  # Cache for embeddings
        self.total_calls = 0
        self.enabled = False
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")
            print("⚠️  Embedding-based detection disabled.")
            self.model = None
            self.enabled = False
        else:
            try:
                print(f"📦 Loading SentenceTransformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.enabled = True
                print(f"✅ Local embedding client initialized (runs offline, no API needed!)")
            except Exception as e:
                print(f"⚠️  Failed to load model {model_name}: {e}")
                self.model = None
                self.enabled = False
    
    def get_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Get embedding for text using local SentenceTransformer model
        
        Args:
            text: Text to embed
            normalize: Normalize embeddings (for cosine similarity)
            
        Returns:
            numpy array of embedding vector (384-dim for all-MiniLM-L6-v2)
        """
        if not self.enabled or self.model is None:
            raise ValueError(
                "Embedding client not enabled. Install sentence-transformers: "
                "pip install sentence-transformers"
            )
        
        # Check cache first
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate embedding using SentenceTransformer
        embedding = self.model.encode(text, normalize_embeddings=normalize, show_progress_bar=False)
        
        # Cache it
        self.embedding_cache[cache_key] = embedding
        self.total_calls += 1
        
        return embedding
    
    def get_embeddings_batch(self, texts: list, normalize: bool = True) -> list:
        """
        Get embeddings for multiple texts in one batch
        
        Args:
            texts: List of texts to embed
            normalize: Normalize embeddings (for cosine similarity)
            
        Returns:
            List of numpy arrays (embeddings)
        """
        if not self.enabled or self.model is None:
            raise ValueError(
                "Embedding client not enabled. Install sentence-transformers: "
                "pip install sentence-transformers"
            )
        
        # Generate embeddings in batch (much faster!)
        # SentenceTransformer.encode() returns a numpy array (n, embedding_dim)
        embeddings_array = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Convert to list of numpy arrays and cache
        embeddings_list = []
        for i, text in enumerate(texts):
            embedding = embeddings_array[i]  # Extract single embedding
            embeddings_list.append(embedding)
            
            # Cache it
            cache_key = hash(text)
            self.embedding_cache[cache_key] = embedding
        
        self.total_calls += 1
        
        return embeddings_list
    
    def flush_batch(self):
        """
        Flush batch (no-op for SentenceTransformers - batching is handled automatically)
        Kept for API compatibility
        """
        pass
    
    def set_patterns_learned(self, count: int):
        """
        Update number of patterns learned (kept for API compatibility)
        No-op for local embeddings - always available!
        """
        if not hasattr(self, 'patterns_learned'):
            self.patterns_learned = 0
        self.patterns_learned = count
    
    def get_stats(self):
        """Get embedding statistics"""
        return {
            "total_calls": self.total_calls,
            "cache_size": len(self.embedding_cache),
            "pending_batch_size": 0,  # No pending batch for SentenceTransformers
            "patterns_learned": getattr(self, 'patterns_learned', 0),
            "model_name": self.model_name,
            "enabled": self.enabled
        }
